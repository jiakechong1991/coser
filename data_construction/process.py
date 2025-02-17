import re 
from typing import List, Tuple
from collections import Counter
import json
import jsonlines
from utils import cached
import traceback
import os
from tqdm import tqdm

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Construct CoSER-style dataset from source books')
    parser.add_argument('--input', type=str, required=True,
                      help='Input jsonl file path containing books data')
    parser.add_argument('--output_dir', type=str, default='data/curated/',
                      help='Output directory path (default: data/curated/)')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of parallel workers (default: 1)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args


@cached
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    logger.info(f"Number of tokens: {num_tokens}")
    return num_tokens


import tiktoken
from utils import get_response, setup_logger, get_response_json, print_json

# Setup logger
logger = setup_logger(__name__, 'process.log')

# Load Anthropic API key from config
with open('config.json', 'r') as f:
    config = json.load(f)

enc = tiktoken.get_encoding("cl100k_base")  # Claude uses cl100k_base encoding

def encode(text):
    return enc.encode(text)

def decode(tokens):
    return enc.decode(tokens)

NEWLINE = enc.encode('\n')[0]

def find_index(lst, key):
    try:
        return lst.index(key)
    except ValueError:
        return -1

@cached
def create_chunk_generator(book, max_tokens):
    # some books may have excessive \t 
    def has_excessive_tabs(content, threshold=0.05):
        tab_count = content.count('\t')
        return (tab_count / len(content)) > threshold
    
    if has_excessive_tabs(book['content']):
        book['content'] = book['content'].replace('\t', '')


    from split import split_book
    chapters = split_book(book)

    results = []

    if not chapters:
        # split the book into chunks, each time get a chunk with max_tokens token
        tokens = encode(book['content'])
        start_index = 0
        
        while start_index < len(tokens):
            if len(tokens) - start_index <= max_tokens:
                chunk = decode(tokens[start_index:])
                results.append(chunk)
                break
            
            chunk_tokens = tokens[start_index:start_index + max_tokens]
            chunk = decode(chunk_tokens)
            results.append(chunk)
            
            start_index += len(chunk_tokens)
    else:

        current_chunk = []
        current_tokens = 0
        
        for chapter in chapters:
            current_chunk.append(chapter['content'])
            current_tokens += len(encode(chapter['content']))
            
            if current_tokens >= max_tokens // 2:
                if current_tokens <= 2 * max_tokens:
                    # Yield the chunk if it's within the desired range
                    results.append(''.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                else:
                    # Split the chunk if it's too large
                    chunk_text = ''.join(current_chunk)
                    chunk_tokens = encode(chunk_text)
                    for i in range(0, len(chunk_tokens), max_tokens):
                        if i + 2 * max_tokens >= len(chunk_tokens):
                            # If we're near the end, return all remaining tokens
                            results.append(decode(chunk_tokens[i:]))
                            break
                        else:
                            results.append(decode(chunk_tokens[i:i + max_tokens]))
                    current_chunk = []
                    current_tokens = 0
        
        # Yield any remaining content
        if current_chunk:
            results.append(''.join(current_chunk))

    # Process results[0] to remove copyright-related lines
    lines = results[0].split('\n')
    filtered_lines = []
    copyright_words = ['rights', 'reserved', 'reproduced', 'copyright', 'reproduce', 'permission']
    
    for line in lines:
        words = line.split()
        if len(words) < 50 and sum(word.lower() in copyright_words for word in words) > 1:
            continue
        filtered_lines.append(line)
    
    results[0] = '\n'.join(filtered_lines)

    return results

import re
import difflib


def ngram_jaccard_similarity(text1, text2, n=3):

    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def jaccard_similarity(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    # Tokenize the texts
    tokens1 = encode(text1)
    tokens2 = encode(text2)
    
    # Generate n-grams
    ngrams1 = set(ngrams(tokens1, n))
    ngrams2 = set(ngrams(tokens2, n))
    
    # Calculate Jaccard similarity
    return jaccard_similarity(ngrams1, ngrams2)

@cached
def find_best_match_passage(candidates, target, n=3, threshold=0.3):
    best_match = None
    best_score = 0

    if isinstance(candidates, list) and isinstance(target, dict) and isinstance(candidates[0], dict):
        target = str(target)
        candidates = [str(c) for c in candidates]

    for i, candidate in enumerate(candidates):
        score = ngram_jaccard_similarity(target, candidate, n)
        if score >= best_score:
            best_score = score
            best_match = i
    
    if best_score >= threshold:
        logger.info(f"Best match: \nInput: {target}\nOutput: {candidates[best_match]}\nScore: {best_score}")
        return best_match
    else:
        return -1


@cached
def find_best_match_sentence(chunk, target, threshold=0.6):
    if target == 'None' or target is None:
        return None

    if isinstance(chunk, str):
        sentences = re.split(r'(?<=[.!?。！？\n])\s*', chunk)
    else: # list
        assert isinstance(chunk, list)  
        sentences = chunk

    best_match = 0
    best_score = 0
    
    for i, sentence in enumerate(sentences):
        # try:
        score = difflib.SequenceMatcher(None, target, sentence).ratio()
        # except:
        #     import pdb; pdb.set_trace()
        if score > best_score:
            best_score = score
            best_match = sentence
    
    logger.info(f"Best match: \nInput: {target}\nOutput: {best_match}\nScore: {best_score}")

    if best_score >= threshold:
        return best_match
    else:
        return None

def process_book(book, max_tokens=8192):

    save_path = f'results/extracted_v3/extracted_data_{book["title"]}.json'
    if os.path.exists(save_path):
        return 

    from utils import set_cache_path
    set_cache_path(f'cache/cache_{book["title"]}.pkl')

    chunk_generator = create_chunk_generator(book, max_tokens)


    results = {
        'chapter_beginnings': [],
        'plots': [],
    }

    remaining_chunk = ''
    truncated_plots = []


    for i, chunk in enumerate(chunk_generator):
        print(f"Processing chunk {i} with {len(encode(chunk))} tokens")

        # Process the chunk
        response = process_book_chunk(book, i, remaining_chunk + chunk, truncated_plots)

        fail_to_parse_responses = []

        if response:
            if isinstance(response, tuple) and len(response) == 3:
                chapter_beginnings, plots, remaining_chunk = response 
            else:
                chapter_beginnings = [] 
                plots = []
                remaining_chunk = ''
                
                fail_to_parse_responses.append(response['fail_to_parse_response'])
                
        else:
            chapter_beginnings = [] 
            plots = []
            remaining_chunk = ''


        # merge plots with previous truncated plots
        for u_plot in truncated_plots:
            # match with the new plots
            idx = find_best_match_passage([p['summary'] for p in plots], u_plot['summary'])

            if idx != -1:
                # merge u_plot into plots[idx]
                plots[idx]['text'] = u_plot['text'] + plots[idx]['text']
  
                old_conversations = u_plot['conversation']
                new_conversations = plots[idx]['conversation']

                merged_conversations = []

                for prev_conv in old_conversations:
                    idx_c = find_best_match_passage([s['scenario'] for s in new_conversations], prev_conv['scenario'])

                    if idx_c != -1:
                        # overlap with the new results, use the new one 
                        merged_conversations.append(new_conversations[idx_c])
                    else:
                        # use the old one 
                        merged_conversations.append(prev_conv)
                
                merged_conversations += [ c for c in new_conversations if c not in merged_conversations ]

                plots[idx]['conversation'] = merged_conversations
 
            else:
                # not matched, just finish this plot 
                u_plot['state'] = 'finished'
                results['plots'].append(u_plot)

        # process plots: finished or truncated
        finished_plots = [plot for plot in plots if plot['state'] == 'finished']
        truncated_plots = [plot for plot in plots if plot['state'] == 'truncated']
        

        results['chapter_beginnings'].extend(chapter_beginnings)
        results['plots'].extend(finished_plots)

    
    for u_plot in truncated_plots:
        # not matched, just finish this plot 
        u_plot['state'] = 'finished'
        results['plots'].append(u_plot)
    
    results['fail_to_parse_responses'] = fail_to_parse_responses
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results

count_nth_generation = {i: 0 for i in range(7)}

def fetch_from_cache(book):

    # open the results/extracted_data_{book["title"]}.json
    with open(f'results/extracted_v1/extracted_data_{book["title"]}.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    save_path = f'results/extracted_v2/extracted_data_{book["title"]}.json'
    #if os.path.exists(save_path):
    #    return 

    import pickle
    # open the cache file 
    with open(f'cache/cache_{book["title"]}.pkl', 'rb') as f:
        cache = pickle.load(f)
    
    keys = [ k for k in cache.keys() if k[0] == 'get_response' ]

    global count_nth_generation

    fail_prompts = []
    responses = {}

    chunk_generator = create_chunk_generator(book, max_tokens=8192)
    chunks = [chunk for chunk in chunk_generator]

    for key, value in cache.items():
        if key[0] == 'get_response':
            dict_string = key[-1][11:-1]

            import ast
            parsed_list = ast.literal_eval(dict_string)

            restored_kwargs = dict(parsed_list)

            if restored_kwargs['model'] == 'claude-3-5-sonnet-20240620' and restored_kwargs['messages'][0]['content'].startswith("\nBased on the provided book chunk, complete the following tasks:\n\n1. Recognize chapter beginnings if"):
                if not restored_kwargs['book']['title'] == book['title']:
                    logger.info(f"Warning: {restored_kwargs['book']['title']} != {book['title']}")
                    continue

                # the value is the response we want 
                # first, count the number of generation
                nth_generation = restored_kwargs['nth_generation']
                count_nth_generation[nth_generation] += 1

                prompt = restored_kwargs['messages'][0]['content']
                responses.setdefault(prompt, {})
                responses[prompt][nth_generation] = value

                if nth_generation == 5:
                    fail_prompts.append(prompt)

    fetched_plots = []

    for prompt in fail_prompts:

        for nth_generation in range(6):
            if nth_generation in responses[prompt]:
                response = responses[prompt][nth_generation]
                #print(f"Nth generation {nth_generation}: {response}")
                required_fields = ["chapter_beginnings", "plots", "chapter_title", "first_sentence", "last_sentence", "summary", "key_characters", "name", "description", "dialogues", "message"]
                if all(field in str(response) for field in required_fields):
                    # This is a candidate but long-and-truncated response
                    from utils import extract_json

                    response = extract_json(response, post_fix_truncated_json=True)

                    if response is None:
                        continue

                    def parse_response(response, chunk, book, **kwargs):
                        if not response:
                            return False
                        
                        try:
                            # sometimes the response is outputted as a list of plots only
                            if (isinstance(response, dict) and 'first_sentence' in response):
                                response = [response]
                            elif isinstance(response, list) and 'first_sentence' in response[0]:
                                response = {
                                    'chapter_beginnings': [],
                                    'plots': response,
                                    'next_chunk_start': None
                                }


                            try:
                                chapter_beginnings = response['chapter_beginnings']
                            except:
                                print(f"Error: {response}")

                            plots = []

                            if response['next_chunk_start']:
                                response['next_chunk_start'] = find_best_match_sentence(chunk, response['next_chunk_start'])
                                if response['next_chunk_start']:
                                    remaining_chunk = chunk[chunk.index(response['next_chunk_start']):]
                                    # keep up to 20% content of the chunk 
                                    remaining_chunk = remaining_chunk[int(len(remaining_chunk) * 0.2):]
                                    # remaining chunk should start with a sentence
                                    remaining_chunk = remaining_chunk[find_index(remaining_chunk, '\n') + 1:]
                                else:
                                    remaining_chunk = ''
                            else:
                                remaining_chunk = ''
                            
                            # Process each plot
                            for unprocessed_plot in response['plots']:
                                
                                chapter_title = unprocessed_plot['chapter_title']


                                # first and last sentence
                                unprocessed_plot['first_sentence'] = find_best_match_sentence(chunk, unprocessed_plot['first_sentence'], threshold=0.6)
                                unprocessed_plot['last_sentence'] = find_best_match_sentence(chunk, unprocessed_plot['last_sentence'], threshold=0.6)

                                first_sentence, last_sentence = unprocessed_plot['first_sentence'], unprocessed_plot['last_sentence']

                                if unprocessed_plot['first_sentence'] and unprocessed_plot['last_sentence']:
                                    original_text = chunk[chunk.index(first_sentence):chunk.index(last_sentence) + len(last_sentence)]
                                else:
                                    original_text = ''

                                plot = {
                                    #'first_sentence': first_sentence,
                                    #'last_sentence': last_sentence,
                                    'text': original_text,
                                    'summary': unprocessed_plot['summary'],
                                    'prominence': unprocessed_plot['prominence'],
                                    'key_characters': unprocessed_plot['key_characters'],
                                    'chapter': chapter_title,
                                    'conversation': unprocessed_plot['conversation'],
                                    'state': unprocessed_plot['state']
                                }

                                plots.append(plot)

                            print_json(response)
                            logger.info(json.dumps(response, ensure_ascii=False, indent=2))

                            return chapter_beginnings, plots, remaining_chunk
                        
                        except Exception as e:
                            logger.error(f"Error processing chunk for book {book['title']}: {e}, {traceback.format_exc()}")
                            return False
                        
                    chunk = prompt.split('==Truncated plot from previous chunk (to be finished)==')[0].split('==Chunk of Book Content==')[-1].strip(' \n')

                    _ = parse_response(response, chunk, book)

                    if _ :
                        chapter_beginnings, plots, remaining_chunk = _
                    else:
                        continue

                    for plot in plots:
                        plot['state'] = 'finished'
                        plot['i_chunk'] = -1
                        #import pdb; pdb.set_trace()
                        for i_chunk, another_chunk in enumerate(chunks):
                            if another_chunk.strip(' \n').endswith(chunk[-100:]):
                                plot['i_chunk'] = i_chunk
                                break


                    fetched_plots.extend(plots)

                    break 
    
    # Now, add the fetched plots to the original results 

    for plot in results['plots']:
        # traverse all the chunks, see which chunk contains the plot['text']
        plot['i_chunk'] = -1
        for i_chunk, chunk in enumerate(chunks):
            if plot['text'][-100:] in chunk:
                plot['i_chunk'] = i_chunk
                break

    # now, merge the new plots with the original plots
    logger.info(f'Number of Original Plots: {len(results["plots"])}, Fetched New Plots: {len(fetched_plots)}, Total Plots: {len(results["plots"]) + len(fetched_plots)}')

    new_plots = results['plots'] + fetched_plots
    new_plots = sorted(new_plots, key=lambda x: x['i_chunk'])

    results['plots'] = new_plots

    # save the results     
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return 

def postprocess_book(book):
    from utils import set_cache_path
    set_cache_path(f'cache/cache_{book["title"]}.pkl')

    save_path = f'results/full/{book["title"]}.json'

    if os.path.exists(save_path):
        return 
    
    logger.info(f"Processing book: {book['title']}")
    with open(f'results/extracted_v2/extracted_data_{book["title"]}.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    plots = results['plots']

    if len(plots) > 0:
        from utils import lang_detect
        language = lang_detect(plots[0]['text'][:100])
        language = {'zh': 'Chinese', 'en': 'English'}.get(language, 'English')
    else:
        language = 'English'


    for plot in plots:
        if isinstance(plot['conversation'], dict) and 'scenario' in plot['conversation']:
            plot['conversation'] = [plot['conversation']]

        for conversation in plot['conversation']:
            # post process the 'scenario' and 'thought' in each conversation 
            input_conversation = {'plot_summary': plot['summary'], 'character_information': plot['key_characters'], **conversation}
            
            conv_key_characters = [ _.get('name', _.get('character', '')) for _ in conversation['key_characters'] if 'name' in _ or 'character' in _]
            prompt = f"""
Given a conversation from {book['title']}, enhance the scene setup and characters' thoughts to create a comprehensive foundation for dramatic performance, i.e., to provide necessary background for actors to act out the conversation:

1. Review the provided conversation and contextual details thoroughly.
2. Expand the 'scenario' with rich situational context that actors need to convincingly perform the scene. Focus on essential background information, while excluding future details to be portrayed in the conversation.
3. Enhance each character's 'motivation' section with their complete mental and emotional state, including their feelings, ideas, objectives, topics they want to discuss, and information they want to convey. Align with their established character and role in the plot. 

===Output Format===
Please provide the output in the following JSON format:
{{
    "scenario": "A detailed scene-setting description that provides actors with essential context and atmosphere (< 200 words). Include all necessary background information while excluding future information to be revealed in the conversation.",
    "key_characters": [
        {{
            "name": "Character name",
            "motivation": "The character's complete mental and emotional state before the conversation (< 100 words). Including their feelings, motivations, objectives, and information they want to convey or discuss."
        }}
    ],
}}

===Requirements===
1. Adhere strictly to the specified output JSON format. 
2. [IMPORTANT] Ensure all DOUBLE QUOTES within all STRINGS are properly ESCAPED, especially when extracting from the text.
3. In the OUTPUT, keep the character names unchanged. 
4. Output in the same language as the input. 
5. Ensure the key_characters include exactly the same characters as the input, including {conv_key_characters}.

===Input Conversation and Background===
{json.dumps(input_conversation, ensure_ascii=False, indent=2)}
"""

            
            from utils import extract_json
            def parse_response(response, characters, **kwargs):
                try:
                    assert 'scenario' in response 
                    assert 'key_characters' in response
                    key_characters = { _['name']: _['thought'] for _ in response['key_characters']}
                    for character in characters:
                        assert character in key_characters
                    return response
                except:
                    return False
            response = get_response_json([extract_json, parse_response], model="claude-3-5-sonnet-20240620", messages=[{"role": "user", "content": prompt}], characters=conv_key_characters, max_retry=5)
            
            try:
                for _ in response['key_characters']:
                    if 'name' not in _ and 'character' in _:
                        _['name'] = _.pop('character')
            except:
                continue
                
            conversation['scenario'] = response['scenario']
            enhanced_thoughts = { _['name']: _['thought'] for _ in response['key_characters']}
            for _ in conversation['key_characters']:
                if 'name' not in _ and 'character' in _:
                    _['name'] = _.pop('character')
                _['thought'] = enhanced_thoughts[_['name']]


    plots = [p for p in plots if all(['name' in c for c in p['key_characters']]) and len(p['key_characters']) == len(set([c['name'] for c in p['key_characters']]))]


    split_index = int(len(plots) * 0.9)


    character_names = set() 
    for plot in plots:
        for character in plot['key_characters']:
            character_names.add(character['name'])
        if isinstance(plot['conversation'], dict) and all([k in plot['conversation'] for k in ['key_characters', 'dialogues']]):
            plot['conversation'] = [plot['conversation']]

        for conversation in plot['conversation']:

            for key_character in conversation['key_characters']:
                try:
                    character_names.add(key_character['name'])
                except:
                    logger.error(f"Error: {key_character}")
                    if 'character' in key_character:
                        key_character['name'] = key_character.pop('character')

            for utterance in conversation['dialogues']:
                try:
                    character_names.add(utterance['character'])
                except:
                    logger.error(f"Error: {utterance}")
                    if 'name' in utterance:
                        utterance['character'] = utterance.pop('name')
    

    character_names = sorted(character_names)

    prompt = """Given a list of character names, titles, or form of address, your task is to: i) generate a list of named characters with their official names (in {language}); ii) For each name in the given list, align it with the official character name if it refers to a named character, or denote it as "impersonal" otherwise.

===Output Format===
Please provide the output in the following JSON format:
{{
    "named_characters": [
        The list of named characters with their official names. Each character should appear only once. 
    ],
    "to_official_name": {{
        "The name in the list": "The official name of the character, or 'impersonal' if it does not refer to a named character."
    }}
}}
===Input===
{character_names}
"""

    prompt = prompt.replace('{character_names}', str(character_names))

    prompt = prompt.replace('{language}', language)

    def parse_response(response, **kwargs):
        if 'named_characters' in response:
            return response
        else:
            return False

    from utils import extract_json
    response = get_response_json([extract_json, parse_response], model="claude-3-5-sonnet-20240620", messages=[{"role": "user", "content": prompt}], max_retry=5)

    try:
        official_names = response['named_characters']
    except:
        response = get_response_json([extract_json, parse_response], model="gpt-4o", messages=[{"role": "user", "content": prompt}], max_retry=5)
        official_names = response['named_characters']
    
    for k, v in response['to_official_name'].items():
        if isinstance(v, list):
            response['to_official_name'][k] = v[0]

    official_names += [ n for n in set(response['to_official_name'].values()) - set(official_names) if n.lower() not in ['impersonal', 'environment'] ]


    to_official_name = response['to_official_name']

    # check if there are names not included in to_official_name
    missing_names = []
    for name in character_names:
        if name not in to_official_name:
            print(f"Warning: {name} not included in to_official_name")
            missing_names.append(name)

    print(f"Missing names: {missing_names}")

    for name in missing_names:
        closest_name = find_best_match_passage(official_names, name)
        print(f"Closest name: {closest_name}")
        if closest_name != -1:
            to_official_name[name] = official_names[closest_name]
        else:
            to_official_name[name] = "impersonal"

    character_datasets = {
        character: {
            "plots": [],
            "conversations": [],
            "utterances": []
        } for character in official_names
    }

    for i_p, plot in enumerate(plots):
        # traverse all plots
        for character in plot['key_characters']:
            # replace character names in key_characters
            if to_official_name[character['name']] != "impersonal":
                character['name'] = to_official_name[character['name']]
                # keep track of involved characters
                character_datasets[character['name']]['plots'].append((i_p, character))


        # then, traver all conversations in a similar way 
        for i_c, conversation in enumerate(plot['conversation']):
            for character in conversation['key_characters']:
                if to_official_name[character['name']] != "impersonal":
                    character['name'] = to_official_name[character['name']]
                    character_datasets[character['name']]['conversations'].append((i_p, i_c, character))
            for i_u, utterance in enumerate(conversation['dialogues']):
                if to_official_name[utterance['character']] != "impersonal":
                    utterance['character'] = to_official_name[utterance['character']]
                    character_datasets[utterance['character']]['utterances'].append((i_p, i_c, i_u, utterance))

    # now     
    prompt = """Please provide a concise, narrative-style character profile for {character_name} from "{book_title}". The profile should read like a cohesive introduction, weaving together the character's background, physical description, personality traits and core motivations, notable attributes, relationships, key experiences, major plot involvement and key decisions or actions, character arc or development throughout the story, and other important details. 
    
The profile should be written in a concise yet informative style, similar to what one might find in a comprehensive character guide, in {language}. Focus on the most crucial information that gives readers a clear understanding of the character's significance in the work. 

You will be provided with summaries and dialogues of some key plots in the book as reference. The profile should be based on either your existing knowledge of the character or the provided information, without fabricating or inferring any inaccurate or uncertain details. 

{character_data}

Now, please generate the character profile, starting with ===Profile===.
"""

    #character_profiles = {}

    for character_name, character_data in character_datasets.items():

        involved_plots = sorted(set([ p[0] for p in character_data['plots']] + [ c[0] for c in character_data['conversations']] + [ u[0] for u in character_data['utterances']]))

        # filter out plots that are not in the training set
        involved_plots = [ i_p for i_p in involved_plots if i_p < split_index]
        
        plot_infos = []

        for i_p in involved_plots:
            plot = plots[i_p]

            plot_info = {
                "plot": plot['summary'],
            }

            for key_character_info in plot['key_characters']:
                if key_character_info['name'] == character_name:
                    plot_info["character_summary"] = key_character_info

            plot_info["conversation"] = [] 

            for conversation in plot['conversation']:
                if character_name in conversation['key_characters'] or character_name in [u['character'] for u in conversation['dialogues']]:
                    plot_info["conversation"].append(conversation)
            
            plot_infos.append(plot_info)

        

        character_prompt = prompt.replace("{character_name}", character_name).replace("{book_title}", book["title"]).replace("{character_data}", json.dumps(plot_infos, ensure_ascii=False, indent=2)).replace("{language}", language)

        print(character_prompt)

        nth_generation = 0
        while True:
            if nth_generation > 0:
                profile = get_response(model="claude-3-5-sonnet-20240620", messages=[{"role": "user", "content": character_prompt}], nth_generation=nth_generation)
            else:
                profile = get_response(model="claude-3-5-sonnet-20240620", messages=[{"role": "user", "content": character_prompt}])
            


            try:
                profile = profile.split("===Profile===", 1)[1].strip() 
                if profile.startswith('I apologize'): profile = ''
                character_datasets[character_name]['profile'] = profile
                break
            except:
                nth_generation += 1
                if nth_generation > 5:
                    character_datasets[character_name]['profile'] = ''
                    break
                continue

    # update the format to make it more readable
    for character_name, character_data in character_datasets.items():
        for i, plot in enumerate(character_data['plots']):
            character_data['plots'][i] = plot[-1]
            character_data['plots'][i]['i_p'] = plot[0]
        
        for i, conversation in enumerate(character_data['conversations']):
            character_data['conversations'][i] = conversation[-1]
            character_data['conversations'][i]['i_p'] = conversation[0]
            character_data['conversations'][i]['i_c'] = conversation[1]
        
        for i, utterance in enumerate(character_data['utterances']):
            character_data['utterances'][i] = utterance[-1]
            character_data['utterances'][i]['i_p'] = utterance[0]
            character_data['utterances'][i]['i_c'] = utterance[1]
            character_data['utterances'][i]['i_u'] = utterance[2]

    results['character_datasets'] = character_datasets
    results['split_plot_index'] = split_index
    # save character profiles to a json file
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


#@cached
def process_book_chunk(book, i_c, chunk, truncated_plots=None):
    logger.info(f"Processing chunk for book: {book['title']}")

    import copy
    if truncated_plots:
        truncated_plots = copy.deepcopy(truncated_plots)
        for plot in truncated_plots:
            plot.pop('text')
    
    prompt = f"""
Based on the provided book chunk, complete the following tasks:

1. Recognize chapter beginnings if they exist in the chunk. Identify the starting sentence of that chapter.
2. Identify the important plots in this chunk. Identify the beginning and ending of each plot by its first and last sentence. Determine the chapter title that the plot belongs to. Set "state" as "truncated" if the plot is truncated in this chunk, or "finished" otherwise. You will be provided with the truncated plots from the previous chunk, and you **must** extend the conversations with the current chunk while keeping the **scenario** unchanged. 
3. Summarize each important plot. For each plot, generate its summary, score its prominence from 1 to 100, and list the key characters and their roles, thoughts and actions in it.
4. Extract conversations for each plot. First, state the scenario and topic of the conversations. Then, list the key characters with their names, descriptions and thoughts at this point. Finally, extract the conversations among them based on the following requirements: 
    i) Ensure the conversations are faithful to the plot and characters. They should be based on the original conversations in the text as much as possible. 
    ii) The conversations should be complete, covering the key dialogues and information. Each conversation should contain at least 10 utterances.
    iii) Each utterance should be composed of one or more thoughts, speech and actions. Use [] outside thoughts, like "[I feel fear and anger, but I cannot show it. I must remain calm and carefully handle his volatile temper.]", which others can't see. Use () outside actions, like "(silence)" or "(smiles at you)," which others can see. Always start an utterance with the character's thought. 
    iv) [IMPORTANT] When generating thoughts, you should think from the characters' perspectives, analyzing the internal thoughts behind their speech and actions in the original text. These thoughts should reflect aspects such as their personal background, personality, values, relationships with others, motivations, and goals. Each thought should be expressed as a phrase or sentence, rather than an adjective or adverb. 
    v) Additionally, describe environmental information (such as scenes, atmosphere, sudden events, etc.) of the conversations as an "utterance" where the "character" field is set as "Environment". The information should exclude characters' active thoughts, observations, and actions.
    vi) Keep the conversation in the same language as the chunk. 
5. Identify the optimal starting point for the subsequent chunk. If the last storyline has been extracted as an truncated plot, set next_chunk_start as None. Otherwise, set next_chunk_start as the first sentence of the last storyline. 

===Output Format===
Please provide the output in the following JSON format:
{{
    "chapter_beginnings": [
        {{
            "beginning_sentence": "Exactly the first line of this chapter (namely the title)."
        }}
    ],
    "plots": [
        // Extend the truncated plots from previous chunk, if any
        {{
            ...
        }}, 
        // New plots in this chunk
        {{
            "chapter_title": "The chapter title that the plot belongs to. Output None if not found.",
            "first_sentence": "Exactly the first sentence of the plot in this **chunk**.",
            "last_sentence": "Exactly the last sentence of the plot in this **chunk**. If the plot is truncated in this chunk, provide the last sentence of this chunk. ",
            "prominence": "Whether this plot is recognized to fans of this book, from 1 to 100.",
            "summary": "The summary of the plot. Just summarize, do not extend unrelated discussions.",
            "key_characters": [
                {{
                    "name": "Character name",
                    "description": "The description of the character before this plot (~20 words).",
                    "summary": "The summary of the character's role, thoughts and behaviors towards this plot, and any significant character development relevant to the plot (~30 words).",
                }}
            ],
            "conversation": [{{
                "scenario": "The scenario at the start of this conversation (providing as much context as possible, but excluding details conveyed in the following conversation)",
                "topic": "The topic of the conversation (~10 words)", 
                "key_characters": [
                    {{
                        "name": "Character name",
                        "motivation": "The thought of the character before starting the conversation, including their attitudes, feelings, motivations, goals, information to convey or topics to be discussed",
                    }}
                ],
                "dialogues": [
                    {{
                        "character": "Character name",
                        "message": "Message, each utterance is composed of thoughts, speech and actions. Use [thought] for internal thoughts, like "[feeling happy]", which others can't see. Use (action) for visible actions, like "(silence)" or "(smiles at you)". Each response starts with the character's internal thought before their speech and actions."
                    }}
                ]
            }}],
            "state": "finished" or "truncated"
        }}
    ],
    "next_chunk_start": "The first sentence of the next chunk."
}}

===Requirements===
1. Adhere strictly to the specified output JSON format. 
2. [IMPORTANT] Ensure all DOUBLE QUOTES within all STRINGS are properly ESCAPED, especially when extracting from the text.
3. In the OUTPUT, use characters' full names, omitting any titles.
4. Maintain Story Fidelity: The plot must accurately reflect the book's content. Avoid introducing plots that are out of context. If the plot contains multiple conversations, prioritize the original dialogue from the book. In the absence of explicit conversations, create dialogue that aligns closely with the plot details.

===Input===

==Book title==
{book['title']}

==Author==
{book['author']}

==Chunk of Book Content== 
{chunk}

==Truncated plot from previous chunk (to be finished)==
{json.dumps(truncated_plots, ensure_ascii=False, indent=2) if truncated_plots else "None"}
"""
    
#     E.g. "[My father's words fill me with awe, but I still feel uneasy.] 
# (Nods seriously, but with a slight frown remaining) 
# I understand, Father. Responsibility is important. But… is killing really necessary? 
# (A flash of compassion in his eyes)
# If someone has done something wrong, can't we give them a chance to make amends?"

    logger.info(prompt)

    def parse_response(response, chunk, book, **kwargs):
        if not response:
            return False
        
        try:
            # sometimes the response is outputted as a list of plots only
            if (isinstance(response, dict) and 'first_sentence' in response):
                response = [response]
            elif isinstance(response, list) and 'first_sentence' in response[0]:
                response = {
                    'chapter_beginnings': [],
                    'plots': response,
                    'next_chunk_start': None
                }


            try:
                chapter_beginnings = response['chapter_beginnings']
            except:
                print(f"Error: {response}")

            plots = []

            if response['next_chunk_start']:
                response['next_chunk_start'] = find_best_match_sentence(chunk, response['next_chunk_start'])

                remaining_chunk = chunk[chunk.index(response['next_chunk_start']):]
                # keep up to 20% content of the chunk 
                remaining_chunk = remaining_chunk[int(len(remaining_chunk) * 0.2):]
                # remaining chunk should start with a sentence
                remaining_chunk = remaining_chunk[find_index(remaining_chunk, '\n') + 1:] 
            else:
                remaining_chunk = ''
            
            # Process each plot
            for unprocessed_plot in response['plots']:
                
                chapter_title = unprocessed_plot['chapter_title']

                # first and last sentence
                unprocessed_plot['first_sentence'] = find_best_match_sentence(chunk, unprocessed_plot['first_sentence'])
                unprocessed_plot['last_sentence'] = find_best_match_sentence(chunk, unprocessed_plot['last_sentence'])

                first_sentence, last_sentence = unprocessed_plot['first_sentence'], unprocessed_plot['last_sentence']

                original_text = chunk[chunk.index(first_sentence):chunk.index(last_sentence) + len(last_sentence)]

                plot = {
                    #'first_sentence': first_sentence,
                    #'last_sentence': last_sentence,
                    'text': original_text,
                    'summary': unprocessed_plot['summary'],
                    'prominence': unprocessed_plot['prominence'],
                    'key_characters': unprocessed_plot['key_characters'],
                    'chapter': chapter_title,
                    'conversation': unprocessed_plot['conversation'],
                    'state': unprocessed_plot['state']
                }

                plots.append(plot)

            print_json(response)
            logger.info(json.dumps(response, ensure_ascii=False, indent=2))

            return chapter_beginnings, plots, remaining_chunk
        
        except Exception as e:
            logger.error(f"Error processing chunk for book {book['title']}: {e}, {traceback.format_exc()}")
            return False
    
    from utils import get_response_json, extract_json

    response = get_response_json([extract_json, parse_response], model="claude-3-5-sonnet-20240620", messages=[{"role": "user", "content": prompt}], book=book, chunk=chunk, fix_truncated_json=True)

    return response

if __name__ == '__main__':
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Read input data
    with jsonlines.open(args.input, mode='r') as reader:
        books_data = list(reader)

    # Clean book titles
    for book in books_data:
        book['title'] = book['title'].replace('/', '-').replace('.', ' ')

    logger.info(f"Processing {len(books_data)} books")

    def process_book_wrapper(book):
        try:
            process_book(book)
            #fetch_from_cache(book)
            result = postprocess_book(book)
            logger.info(f"Successfully processed book: {book.get('title', 'Unknown')}")
            return result
        except Exception as e:
            logger.error(f"Error processing book {book.get('title', 'Unknown')}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    if args.num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        

        logger.info(f"Starting parallel processing with {args.num_workers} workers")

        # Process books in parallel
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            processed_books = list(tqdm(
                executor.map(process_book_wrapper, books_data),
                total=len(books_data),
                desc="Processing books"
            ))
    else:
        processed_books = []
        for book in tqdm(books_data):
            processed_book = process_book_wrapper(book)
            processed_books.append(processed_book)

    # Filter out failed processes
    successful_books = [book for book in processed_books if book is not None]
    logger.info(f"Successfully processed {len(successful_books)}/{len(books_data)} books")














