import json
from typing import Dict, List
from utils import config, get_response, lang_detect, remove_inner_thoughts, load_json, USER, fix_repeation
from prompts import roleplay_prompts as prompts
from openai import OpenAI
import copy
import json

from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS  
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from utils import setup_logger
logger = setup_logger(__name__, 'agent.log')

STEPCAST_ROUTER = "http://stepcast-router:9200"
ENVIRONMENT = "Environment"
NSP = "NSP"

special_characters = [ENVIRONMENT, NSP]

class BGE_M3_Embedding(Embeddings):
    def __init__(self):
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=f"{STEPCAST_ROUTER}/v1"
        )
        self.model = "eval-BAAI-bge-m3-embedding"
        self.embedding_ctx_length = 8192  

    def _embed(self, text: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint."""
        if not isinstance(text, str):
            raise ValueError(f"Expected string input, got {type(text)}")
        
        text = text.replace("\n", " ")
        try:
            embedding = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            return embedding.data[0].embedding
        except Exception as e:
            print(f"Error during embedding: {e}")
            print(f"Problematic text: {text[:100]}...")  
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self._embed(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = []
        for i in range(0, len(texts), 100):  
            batch = texts[i:i+100]
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                embeddings.extend([data.embedding for data in response.data])
            except Exception as e:
                print(f"Error during batch embedding: {e}")
                raise
        return embeddings

def build_rag_corpus(character, database, target):
    # build the rag corpus from the database 
    # considering four targets: book, summary, conversation, mixed
    
    if database is None:
        return None

    book_corpus = []
    summary_corpus = []
    conversation_corpus = []
    
    for plot in database['detailed_plots']:
        # on average, each plot text is about 1000-3000 tokens, so we do not split it.        
        book_corpus.append(plot['text'])
        

        # summary_corpus.append(json.dumps(summary_info, indent=2, ensure_ascii=False))
        character_summary = { _['name']: _['summary'] for _ in  plot['key_characters']}.get(character, '')

        if character_summary:
            character_summary = f"{character}'s role: " + character_summary

        summary_corpus.append('PLOT: ' + plot['summary'] + '\n' + character_summary)

        conversation_info = {"summary": plot['summary'], "conversation": copy.deepcopy(plot['conversation'])}
        for conversation in conversation_info['conversation']:

            for character_info in conversation['key_characters']:
                character_info.pop("i_p", None)
                character_info.pop("i_c", None)
            
            for dialogue in conversation['dialogues']:
                dialogue.pop("i_p", None)
                dialogue.pop("i_c", None)
                dialogue.pop("i_u", None)

            
            from utils import conversation_to_str
            try:
                conversation_corpus.append(conversation_to_str(conversation=conversation['dialogues'], background={'Plot Background': plot['summary'], 'Scenario': conversation['scenario'], 'topic': conversation['topic']}))
            except Exception as e:
                # print error information 
                from utils import setup_logger
                logger.error(f'Error in conversation_to_str: {e}')
                logger.error(f'Conversation: {conversation}')


    
    # Average number of tokens in book corpus: 2589.40
    # Average number of tokens in summary corpus: 368.40
    # Average number of tokens in conversation corpus: 1141.40
    
    corpus_map = {
        'book': [(book_corpus, 1, 'book')],
        'summary': [(summary_corpus, 1, 'summary')],
        'summary-3': [(summary_corpus, 3, 'summary')],
        'conversation': [(conversation_corpus, 1, 'conversation')],
        'mixed': [(summary_corpus, 3, 'summary'), (conversation_corpus, 1, 'conversation')],
        'all_exp_conv': [(summary_corpus, 10, 'summary'), (conversation_corpus, 1, 'conversation')
                         ]
    }

    corpora = corpus_map[target]
    
    retriever = {}

    for (corpus, k, target_type) in corpora:
        # Create documents
        documents = [Document(page_content=doc) for doc in corpus]

        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        for i, doc in enumerate(split_docs):
            doc.metadata['idx'] = i

        # Initialize custom embedding model
        custom_embed_model = BGE_M3_Embedding()
        

        # Create vector store by adding docs one by one to handle potential errors
        try:
            vectorstore = FAISS.from_documents(split_docs, custom_embed_model)
        except Exception as e:
            print(f"Cannot create vectorstore at once: {e}; will try again.")
            # Initialize empty vectorstore with first document
            try:
                vectorstore = FAISS.from_documents([split_docs[0]], custom_embed_model)
                for doc in split_docs[1:]:
                    vectorstore.add_documents([doc])
                
                print('Successfully created vectorstore')
            except:
                # create an empty vectorstore
                continue
        
        if k != -1:
            retriever[target_type] = vectorstore.as_retriever(search_kwargs={"k": k})
        else:
            class SequentialRetriever:
                def __init__(self, docs):
                    self.docs = docs
                
                # Add any other interface methods that might be needed
                def invoke(self, query):
                    return self.docs

            # Store original documents and create retriever
            retriever[target_type] = SequentialRetriever(split_docs)

            #retriever[target_type] = vectorstore.as_retriever(search_kwargs={"k": len(split_docs)})
        

    return retriever

def rag(contexts, retriever, target_type):
    # This function should return relevant information from a database, based on the input contexts

    title_header = {'book': '====Book Content====\n\n', 'summary': '====Historical Experience====\n\n', 'conversation': '====Historical Conversation====\n\n'}[target_type]
    title = {'book': 'Content', 'summary': 'Historical Experience', 'conversation': 'Historical Conversation'}[target_type]

    query = "\n\n".join([msg["content"] for msg in contexts])
    retrieved_docs = retriever.invoke(query)

    if retrieved_docs and 'idx' in retrieved_docs[0].metadata:
        #sort by index (time order)
        retrieved_docs = sorted(retrieved_docs, key=lambda x: x.metadata['idx'])

    if len(retrieved_docs) > 1:
        # concat with index
        relevant_info = title_header + ''
        for i, doc in enumerate(retrieved_docs):
            relevant_info +=  f'{title} {i+1}\n' + doc.page_content + '\n\n'
    else:
        relevant_info = title_header + "\n\n".join([doc.page_content for doc in retrieved_docs])

    return relevant_info

class Agent:
    def __init__(self, model: str, name, database: Dict, system_prompt: str = None, scene: Dict = None, retrieval_target: str = 'conversation'):
        self.model = model 
        self.name = name 
        self.database = database
        self.scene = scene


        self.system_prompt = system_prompt if system_prompt else get_system_prompt(name, scene)
        
        # for baseline models without role-playing fine-tuning, add more prompt 
        # if not ('gpt' or 'claude' or 'roleplay' in model name)

        self.system_prompt = self.system_prompt.strip('\n')
        
        if retrieval_target and database:
            self.retrievers = build_rag_corpus(name, database, retrieval_target)
        else:
            self.retrievers = None

        if self.name not in special_characters:
            self.system_prompt = self.system_prompt + '\n\nSpeak concisely as humans, instead of being verbose. Limit your response to 60 words.\n\n'

        if self.model in ['llama3p1-8b-instruct', 'llama31-70-chat'] or 'step' in self.model:
            self.system_prompt = self.system_prompt + f'Start your response with "{name}: ". Avoid speaking as other characters.\n\n'
        
        if self.model.startswith('claude') or self.model.startswith('o1'):
            self.system_role = 'user'
        else:
            self.system_role = 'system'

        self.messages = [{"role": self.system_role, "content": self.system_prompt}]
        

    def chat(self) -> str:
        try:
            # print(self.messages)
            # print(len(self.messages))

            messages = self.messages
            if self.retrievers:
                # retrieve relevant information from the corpus
                # the context is the last three messages, except the system prompt
                contexts = self.messages[1:]
                contexts = contexts[-3:]

                knowledge = ''
                for target_type, retriever in self.retrievers.items():
                    knowledge += rag(contexts, retriever, target_type)

                # add rag into system prompt
                messages = copy.deepcopy(self.messages)
                messages[0]['content'] = messages[0]['content'].replace('{retrieved_knowledge}', '<begin of background information>\n\n' + knowledge + '\n\n<end of background information>\n\n')

            

            from utils import get_response_with_retry
            response = get_response_with_retry(model=self.model, messages=messages, max_tokens=512)

            # Parse the response to extract only the utterance of self.name character
            def parse_response(response: str, character_name: str) -> str:
                lines = response.split('\n')

                current_character = None
                current_utterance = ""

                parsed_utterances = []

                for line in lines:
                    # Check if the line starts with a character name
                    if ':' in line:
                        character = line.split(':', 1)[0].strip()
                        
                        if current_character != character:
                            # New character speaking
                            if current_utterance:
                                parsed_utterances.append((current_character, current_utterance))
                            current_character = character
                            current_utterance = ""

                    # Continue current character's utterance
                    current_utterance += line + "\n"
            
                # Handle the last utterance
                if current_utterance:
                    parsed_utterances.append((current_character, current_utterance))
                
                parsed_utterances = [utterance for character, utterance in parsed_utterances if character == character_name][0]

                # print('Original response: ', response)
                # print('Parsed response: ', parsed_utterances)

                return parsed_utterances

            # Parse the response to keep only the utterance of self.name character
            if (self.model in ['llama3p1-8b-instruct', 'llama31-70-chat'] or 'step' in self.model) and self.name != 'NSP':
                response = parse_response(response, self.name)
            
            if not any(model_type in self.model for model_type in ['gpt', 'claude']) and self.name != 'NSP':
                # if response contains repetition, fix it
                _ = fix_repeation(response)
                if _: 
                    logger.info(f'{self.model} Repetition found and fixed: {response} {_}')
                    response = _

            return response

        except Exception as e:
            import traceback
            print(f"Error getting response: {e}")
            traceback.print_exc()
            
            return ""
    
    def update(self, role: str, message: str):
        if message:
            # message could be '' empty, which means the first speech, starting from only system prompt 
            if self.messages and self.messages[-1]['role'] == role:
                self.messages[-1]['content'] = self.messages[-1]['content'] + '\n\n' + message
            else:
                self.messages.append({"role": role, "content": message})

        return

    def reset(self):
        self.messages = self.messages[:1]
    
        

if __name__ == '__main__':
    pass
