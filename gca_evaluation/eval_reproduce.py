# evaluate whether an role-playing agent can reproduce the same conversation as the original character 
import json 
from utils import get_response
import argparse
from tqdm import tqdm
from utils import setup_logger
from agent import Agent
import random
import os
from utils import remove_inner_thoughts
from utils import get_environment_prompt, get_nsp_prompt
from utils import get_response_json, extract_json

random.seed(42)

logger = None

parser = argparse.ArgumentParser(description='Evaluate role-playing agent reproduction')
parser.add_argument('--actor_model', type=str, default='claude35sonnet', help='Name of the model to use for role-playing')
parser.add_argument('--judge_model', type=str, default='gpt-4o', help='Name of the model to use for judging')
parser.add_argument('--actor_retrieval', action='store_true', help='Enable memory retrieval for actor model')
parser.add_argument('--judge_retrieval', action='store_true', help='Enable memory retrieval for judge model')
parser.add_argument('--retrieval_target', type=str, default='conversation', choices=['book', 'summary', 'summary-3', 'conversation', 'mixed'], help='Target for retrieval')
parser.add_argument('--regenerate', action='store_true', help='Regenerate the simulation results')
parser.add_argument('--reevaluate', action='store_true', help='Reevaluate the simulation results')
parser.add_argument('--n_samples', type=int, default=10000, help='Number of samples to evaluate')
parser.add_argument('--continue_from', type=int, default=0, help='For each conversation, continue from the i-th message')
parser.add_argument('--wo_thought', default=False, action='store_true', help='Disable inner thoughts in generation')
parser.add_argument('--nth_exp', type=int, default=0, help='The experiment repeated in the nth time')
args = parser.parse_args()

book_dataset_path = 'processing/results/full'

ENVIRONMENT = 'Environment'
NSP = "NSP"
special_characters = [NSP, ENVIRONMENT]
env_model = 'llama3-1210-8epc'


def transform_conversations(dataset):
    role_map = {
        'human': 'user',
        'assistant': 'assistant',
        'system': 'system'
    }

    # traverse the conversations in this dataset, transforming the format by
    # 1. replacing the keys 'from' and 'value' with 'role' and 'content'
    # 2. unfolding the conversation from the dict format into list format

    new_dataset = [] 

    for book_title, conversations in dataset.items():
        for conversation in conversations:
            for message in conversation:
                message['role'] = role_map[message.pop('from').lower()]
                message['content'] = message.pop('value')

            new_dataset.append((book_title, conversation))
    
    return new_dataset


def generate_self_play_conversations(test_dataset_path, actor_model, actor_retrieval, retrieval_target, nth_exp=0):
    random.seed(42)

    from utils import set_cache_path
    cache_path = f'cache/cache_{actor_model}{"_rag=" + retrieval_target if actor_retrieval else ""}.pkl'
    if nth_exp > 0:
        cache_path = f'{cache_path}-repeat={nth_exp}'

    set_cache_path(cache_path)
    
    test_dataset = json.load(open(test_dataset_path, 'r'))
    results = []

    # Extract filename without extension from test_dataset_path
    simulation_path = f'./exp/results/{exp_name}-{os.path.splitext(os.path.basename(test_dataset_path))[0]}_{actor_model}{"_rag=" + retrieval_target if actor_retrieval else ""}.json'

    # if simulation_path exists, load it
    if os.path.exists(simulation_path) and not args.regenerate:
        return json.load(open(simulation_path, 'r'))

    if args.reevaluate: 
        # assume the simulation results are already generated 
        raise ValueError(f'Simulation result {simulation_path} not exists')


    # generate self-play conversations
    for circumstance in test_dataset[:args.n_samples]:
        book_title = circumstance['book']
        plot = circumstance['plot']
        i_p = plot['i_p'] 
        conversation = circumstance
        i_c = conversation['i_c']
        character_profiles = circumstance['character_profiles']

        logger.info(f'==========Book {book_title}==========')

        if actor_retrieval:
            book_database = json.load(open(f'{book_dataset_path}/{book_title}.json', 'r'))

        plot_characters = [ c['name'] for c in plot['key_characters']] 

        speaking_characters_w_env = conversation['speaking_characters_w_env']
        if ENVIRONMENT not in speaking_characters_w_env:
            speaking_characters_w_env.append(ENVIRONMENT)
        major_characters = conversation['major_characters']

        character_agents = {}

        involved_character_profiles = {}

        for character in speaking_characters_w_env:    
            if character == ENVIRONMENT:
                continue
            else:
                character_profile = character_profiles.get(character, '')

                if character in plot_characters:
                    character_info = [c for c in plot['key_characters'] if c.get('name', '') == character][0]

                    if 'description' in character_info:
                        character_profile = character_info.get('description', '').strip('\n') + '\n\n' + character_profile.strip('\n')
                    
                character_profile = character_profile.strip(' \n')

                if character_profile != '':
                    involved_character_profiles[character] = character_profile

        # Initiate an agent for each speaking character 
        for character in speaking_characters_w_env + [NSP]:    
            # Create agents for the characters
            
            if character == NSP:
                system_prompt = get_nsp_prompt(speaking_characters_w_env, conversation['scenario'])
                character_database = None
            elif character == ENVIRONMENT:
                system_prompt = get_environment_prompt(major_characters, conversation['scenario'])
                character_database = None
            else:
                if actor_retrieval and character in book_database['character_datasets']:
                    character_database = book_database['character_datasets'][character]
                    
                    involved_plots = [_['i_p'] for _ in character_database['plots']] + [_['i_p'] for _ in character_database['conversations']] + [_['i_p'] for _ in character_database['utterances']]

                    involved_plots = sorted(set(involved_plots))
                
                    character_database['detailed_plots'] = [ book_database['plots'][i] for i in involved_plots ] 

                else:
                    character_database = None

                character_profile = involved_character_profiles.get(character, '')

                if character in plot_characters:
                    character_info = [c for c in plot['key_characters'] if c.get('name', '') == character][0]

                    # do not add character summary into prompt 

                character_profile = character_profile.strip(' \n')

                find_thought = [ c.get('thought', '') for c in conversation['key_characters'] if c.get('name', '') == character]
                if find_thought: 
                    thought = find_thought[0]
                else:
                    thought = ''

                from utils import get_character_prompt
                if 'llama' in actor_model and actor_model not in ['llama31-70-chat', 'llama3-1210-8epc-wo-cot', 'llama3p1-8b-instruct', 'llama3p1-8b-1211-wo-cot-8epc']:

                    logger.info('SFT tuned model')
                    add_output_example = False
                else:
                    # ['qwen2.5-72b-instruct', 'Qwen2-7B-Instruct', 'o1-mini', 'gpt3.5', 'abab7-preview-chat', 'Doubao-pro-32k', 'claude-3-haiku']
                    add_output_example = True

                system_prompt = get_character_prompt(book_title, character, character_profile, plot["summary"], conversation["scenario"], thought, thoughtless=args.wo_thought, other_character_profiles=involved_character_profiles, exclude_plot_summary=True, fixed_template=True, add_output_example=add_output_example, add_rag=actor_retrieval)
            
            #logger.info(system_prompt)

            if character not in special_characters:
                character_model = actor_model
            elif character == ENVIRONMENT:
                character_model = 'gpt-4o'
            elif character == NSP:
                character_model = 'llama3-1210-8epc'  
            else:
                raise ValueError(f'Invalid character: {character}')

            character_agent = Agent(character_model, character, character_database, system_prompt=system_prompt, retrieval_target=retrieval_target if (actor_retrieval and character not in special_characters) else None)

            character_agent.update('user', "===Conversation Start===\n\n")

            character_agents[character] = character_agent
            

        # Generate Agent Conversation
        max_rounds = 20
        agent_conversations = []
            
        # Set the initial speaker as the first speaker in the dialogue
        current_speaker = speaking_characters_w_env[0]
        
        for i_round in range(max_rounds):
            if current_speaker == "<END CHAT>":
                break

            # Current speaker generates a response
            logger.info(f'===Round {i_round}===\n')
            for actor in [current_speaker, "NSP"]:

                current_agent = character_agents[actor]

                from utils import add_speaker_name
                
                if args.continue_from > i_round:
                    # use the message from ground truth
                    if actor == current_speaker:
                        response = conversation['dialogues'][i_round]['message']
                    else:
                        # 'NSP'
                        if i_round < len(conversation['dialogues']) - 1:
                            response = conversation['dialogues'][i_round+1]['character']
                        else:
                            response = '<END CHAT>'
                else:
                    response = current_agent.chat()

                if actor == "NSP":
                    # predict the next speaker
                    next_actor = response
                    if ':' in next_actor:
                        # the nsp model overgenerate the response
                        next_actor = next_actor.split(':')[0].strip()

                    if next_actor == "<END CHAT>" and i_round >= 5:
                        current_speaker = "<END CHAT>"
                    elif next_actor in speaking_characters_w_env and next_actor != current_speaker:
                        current_speaker = next_actor
                    else:
                        candidates = set(major_characters + [ENVIRONMENT] ) - {current_speaker}
                        if len(candidates) == 0:
                            candidates = set(speaking_characters_w_env) - {current_speaker}
                        current_speaker = random.sample(candidates, 1)[0]  
                    logger.info(f"Next speaker: {current_speaker} (Raw response: {response})")

                    agent_conversations.append({"role": actor, "content": next_actor})

                    current_agent.update('assistant', next_actor)
                
                else:
                    # let the character speak (including the environment)
                    response = add_speaker_name(response, actor)
                    if actor == ENVIRONMENT:
                        logger.info(f"{env_model}: {response}\n")
                    else:
                        logger.info(f"{actor_model}: {response}\n")

                    agent_conversations.append({"role": actor, "content": response})

                    for other_actor, other_agent in character_agents.items():
                        if other_actor == actor:
                            other_agent.update('assistant', response)
                        else:
                            other_agent.update('user', remove_inner_thoughts(response))

        results.append({
            'book_title': book_title,
            'i_p': i_p,
            'i_c': i_c,
            'circumstance': circumstance,
            'simulation': agent_conversations,
            'involved_character_profiles': involved_character_profiles
        })

    # save the results
    os.makedirs(os.path.dirname(simulation_path), exist_ok=True)
    with open(simulation_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results

global_results_by_case = {} 

def evaluate_self_play_conversations(test_dataset_path, actor_model, actor_retrieval, judge_model, judge_retrieval, retrieval_target, nth_exp=0):
    from utils import set_cache_path

    cache_path = f'cache/cache_{actor_model}{"_rag=" + retrieval_target if actor_retrieval else ""}.pkl'
    if nth_exp > 0:
        cache_path = f'{cache_path}-repeat={nth_exp}'
    set_cache_path(cache_path)

    # load the test dataset
    with open(test_dataset_path, 'r') as f:
        test_dataset = json.load(f)
    
    actor_setting = f'{actor_model}{"_rag=" + retrieval_target if actor_retrieval else ""}'

    simulation_path = f'./exp/results/{exp_name}-{os.path.splitext(os.path.basename(test_dataset_path))[0]}_{actor_setting}.json'

    evaluation_path = f'./exp/results/{exp_name}-{os.path.splitext(os.path.basename(test_dataset_path))[0]}_{actor_setting}_eval.json'
    

    if os.path.exists(evaluation_path) and not (args.regenerate or args.reevaluate):
        res = json.load(open(evaluation_path, 'r'))
        return res['avg_scores'], res['results_by_case']
    
    # load the simulation results
    simulation_results = json.load(open(simulation_path, 'r'))
    
    count_input_tokens = 0 
    count_output_tokens = 0 
    count_samples = 0 

    dimensions = ['Storyline Consistency', 'Anthropomorphism', 'Character Fidelity', 'Storyline Quality']

    scores = {
        d: [] for d in dimensions
    }

    results_by_case = {'groundtruth': {}, actor_setting: {}}

    actor_generated_text_list = []

    for result in simulation_results:
        book_title = result['book_title']
        circumstance = result['circumstance']
        plot = circumstance['plot']
        i_p = plot['i_p'] 
        conversation = circumstance
        i_c = conversation['i_c']

        simulation = result['simulation']
        
        logger.info(f'book {book_title}')

        reference = conversation['dialogues']

        simulation = [m for m in simulation if m['role'] != NSP]

        orig_simulation = simulation

        #remove the inner thoughts for fair comparison
        simulation = [ m if m['role'] == ENVIRONMENT else 
            {**m, 'content': remove_inner_thoughts(m['content'])} 
            for m in simulation  ]

        actor_generated_text = '\n\n'.join([remove_inner_thoughts(m['content']).strip('\n') for m in simulation if m['role'] != ENVIRONMENT])

        actor_rounds = len([m for m in simulation if m['role'] != ENVIRONMENT])
        from utils import tokenize_words
        actor_word_count = len(tokenize_words(actor_generated_text))

        actor_generated_text_list.append(actor_generated_text)

        reference = [ m if m['character'] == ENVIRONMENT else 
            {**m, 'message': remove_inner_thoughts(m['message'])} 
            for m in reference  ]

        simulation_text = '\n\n'.join([m['content'].strip('\n') for m in simulation])
        reference_text = '\n\n'.join([f"{m['character']}: {m['message']}".strip('\n') for m in reference])

            
        logger.info(f'===Simulation of {actor_setting}===\n\n**************\n{simulation_text}\n\n**************\n\n===Reference===\n\n**************\n{reference_text}\n\n**************\n\n')

        scenario_text =  conversation['scenario']

        character_profile_text = '\n\n'.join([f"### {character}\n\n{profile.strip('')}" for character, profile in result['involved_character_profiles'].items()])

        additional_instructions = ''
        if args.continue_from > 0:
            additional_instructions = f'Please note that the first {args.continue_from} messages in the simulated conversation are the same as the reference. Focus your evaluation only on the content after these messages.'

        major_characters = conversation['major_characters']

        input = simulation_text 

        def parse_response(response, **kwargs):
            try:
                assert isinstance(response, dict)
                for k, v in response.items():
                    assert k in dimensions
                    assert 'flaws' in v

                    for f in v['flaws']:
                        if f.get('severity', None) is None:
                            f['severity'] = 1


                return response
            except:
                return False

        logger.info(f'{book_title}-{i_p}-{i_c}-{scenario_text}')

        eval_result = {}

        for dimension in dimensions:
            from prompts import critic_prompts
            critic_prompt = critic_prompts['self-play-deduct-template'].replace('{book}', book_title).replace('{plot_summary}', plot['summary']).replace('{scenario}', scenario_text).replace('{character_profiles}', character_profile_text).replace('{original_conversation}', reference_text).replace('{major_characters}', ', '.join(major_characters)).replace('{additional_instructions}', additional_instructions).replace('{dimension_name}', dimension).replace('{dimension_brief}', critic_prompts['dimension_details'][dimension]['dimension_brief']).replace('{dimension_criteria}', critic_prompts['dimension_details'][dimension]['dimension_criteria'])

            res = get_response_json([extract_json, parse_response], model=judge_model, messages=[{"role": "system", "content": critic_prompt}, {"role": "user", "content": input}])
            logger.info(json.dumps(res, ensure_ascii=False, indent=2)) 
            eval_result.update({dimension: res[dimension]})
            
        
            
            res[dimension]['score'] = max(0, 100 - (sum([f['severity'] for f in res[dimension]['flaws'] if isinstance(f['severity'], int)]) - 0.3 * actor_rounds) * 5)


        eval_result['score'] = sum([eval_result[dimension]['score'] for dimension in dimensions]) / len(dimensions)


        results_by_case['groundtruth'][f'{book_title}-{i_p}-{i_c}'] = {'scenario': scenario_text, 'groundtruth': reference_text, 'score': 100}

        results_by_case[actor_setting][f'{book_title}-{i_p}-{i_c}'] = {
            'simulation': orig_simulation,
            'simulation_text': simulation_text,
            'score': eval_result['score'],
            'critique': eval_result,
        }

        global all_model_results_str
        all_model_results_str += f"{book_title}-{i_p}-{i_c}-{scenario_text}\n{json.dumps(eval_result, ensure_ascii=False, indent=2)}\n\n"           

        from utils import num_tokens_from_string
        count_input_tokens += num_tokens_from_string(critic_prompt + input)
        count_output_tokens += num_tokens_from_string(str(eval_result))
        count_samples += 1
    
        for dimension in dimensions:
            scores[dimension].append(eval_result[dimension]['score'])


    avg_scores = {dimension: sum(scores[dimension]) / max(1, len(scores[dimension])) for dimension in dimensions }

    avg_scores['avg'] = sum(avg_scores.values()) / len(avg_scores)

    avg_scores = {dimension: round(avg_scores[dimension], 3) for dimension in avg_scores }

    from utils import avg, ttr

    avg_scores['ttr_single'] = round(avg([ttr(text) for text in actor_generated_text_list]), 3)
    avg_scores['ttr_all'] = round(ttr(' '.join(actor_generated_text_list)), 3)

    logger.info(f'{actor_model}{"_rag=" + retrieval_target if actor_retrieval else ""}: Average score of {count_samples} samples: \n{avg_scores["avg"]} {avg_scores} on {test_dataset_path}')

    all_model_results_str += f'{actor_model}{"_rag=" + retrieval_target if actor_retrieval else ""}: Average score of {count_samples} samples:\n{avg_scores["avg"]} {avg_scores} on {test_dataset_path}\n'

    # save the evaluation results
    with open(evaluation_path, 'w') as f:
        json.dump({'avg_scores': avg_scores, 'results_by_case': results_by_case}, f, ensure_ascii=False, indent=2)

    return avg_scores, results_by_case

from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from rouge import Rouge
import nltk

def calculate_bleu_rouge(reference, candidate):
    try:
        
        reference_tokens = word_tokenize(reference.lower())
        candidate_tokens = word_tokenize(candidate.lower())
        
        bleu_score = sentence_bleu([reference_tokens], candidate_tokens)
        
        rouge = Rouge()
        
        rouge_scores = rouge.get_scores(candidate, reference)[0]
        
        rouge_1 = rouge_scores['rouge-1']['f']
        rouge_2 = rouge_scores['rouge-2']['f']
        rouge_l = rouge_scores['rouge-l']['f']
        
        return {
            'bleu': bleu_score,
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l
        }
    
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return None
    
def evaluate_bleu_rouge(results_by_case, continue_from=0):
    samples = results_by_case['groundtruth'].keys()

    model_name = list(results_by_case.keys())[1]

    results_list = [] 

    for sample in samples:
        reference_text = results_by_case['groundtruth'][sample]['groundtruth']
        simulation_text = results_by_case[model_name][sample]['simulation_text']

        if continue_from > 0:
            reference_paragraphs = reference_text.split('\n\n')
            simulation_paragraphs = simulation_text.split('\n\n')

            for i in range(len(reference_paragraphs)):
                if reference_paragraphs[i] != simulation_paragraphs[i].replace('  ', ' '):
                    break
            
            reference_text = '\n\n'.join(reference_paragraphs[i:])
            simulation_text = '\n\n'.join(simulation_paragraphs[i:])

        results = calculate_bleu_rouge(reference_text, simulation_text)
        results_list.append(results)
    
    # average results_list
    avg_results = {}
    for key in results_list[0]:
        avg_results[key] = sum([results[key] for results in results_list]) / len(results_list)

    return avg_results

def merge_global_results(global_results, results):
    #global_results_by_case.update(results)

    for model_args in results:
        if model_args not in global_results:
            global_results[model_args] = results[model_args]
        else:
            # append 
            for circumstance in results[model_args]:
                global_results[model_args][circumstance] = results[model_args][circumstance]

    return global_results

if __name__ == "__main__":
    
    # Open the two datasets
    # Traverse the character's utterance
    # Generate a response representing the character, and compare it with the original utterance

    
    if 1:
        test_scene_path = 'processing/results/final/rp_1220'


        test_dataset_paths = [
            f'{test_scene_path}/test_circumstance_id.json',
            f'{test_scene_path}/test_circumstance_ood.json'
        ]

        all_model_results_str = ''


        if args.nth_exp >= 0:
            nth_exps = [args.nth_exp]
        else:
            repeat_times = 3
            nth_exps = range(repeat_times)

        for nth_exp in nth_exps:

            
            print(f'Repeat the experiment for the {nth_exp} time')

            exp_name = f'1221-continue_from={args.continue_from}-200case-env_model=gpt-4o-nsp=llama3-1210-8epc'
                
            if nth_exp > 0:
                exp_name += f'-repeat={nth_exp}'

            logger = setup_logger(__name__, './' + exp_name + '.log')

            logger.info(f'Repeat {nth_exp} times')
 
            final_results = {}

            global_results_by_case = {} 

            for test_dataset_path in test_dataset_paths:
                split = test_dataset_path.replace('.json', '').split('/')[-1].split('_')[-1]
                
                model_results = {} 

                from concurrent.futures import ProcessPoolExecutor
                import functools

                def generate(model_args):
                    test_dataset_path, actor_model, args, rag_setting, nth_exp = model_args
                    logger.info(f'Generating {actor_model}{"_rag=" + rag_setting["retrieval_target"] if rag_setting["actor_retrieval"] else ""} on {test_dataset_path}')
                    results = generate_self_play_conversations(
                        test_dataset_path,
                        actor_model, 
                        rag_setting['actor_retrieval'],
                        rag_setting['retrieval_target'],
                        nth_exp
                    )

                    return results

                def evaluate(model_args):
                    test_dataset_path, actor_model, args, rag_setting, nth_exp = model_args
                    logger.info(f'Evaluating {actor_model}{"_rag=" + rag_setting["retrieval_target"] if rag_setting["actor_retrieval"] else ""} on {test_dataset_path}')

                    scores, results_by_case = evaluate_self_play_conversations(
                        test_dataset_path,
                        actor_model,
                        rag_setting['actor_retrieval'],
                        args.judge_model,
                        args.judge_retrieval,
                        rag_setting['retrieval_target'],
                        nth_exp
                    )

                    bleu_rouge_scores = evaluate_bleu_rouge(results_by_case, continue_from=args.continue_from)

                    scores.update(bleu_rouge_scores)

                    return {f'{actor_model}{"_rag=" + rag_setting["retrieval_target"] if rag_setting["actor_retrieval"] else ""}': scores}, results_by_case

                def generate_and_evaluate(model_args):
                    results = generate(model_args)
                    scores, results_by_case = evaluate(model_args)
                    return scores, results_by_case
                
                models = [ 'gpt3.5', 'gpt-4o', 'gpt-4o-mini', 
                    'llama3-1210-8epc', 'llama31-70-chat', 
                    'abab7-preview-chat', 'Doubao-pro-32k', 'doubao-1.5-pro-32k', 
                    'claude35sonnet', 'claude-3-haiku', 
                    #'llama3p1-8b-1211-8epc', 
                    'llama3p1-8b-0119',
                    'llama3p1-8b-instruct',
                    #'qwen2.5-72b-instruct', 
                    'Qwen2-7B-Instruct',
                    'gemini-pro'
                ]

                models += ['custom-Higgs-Llama-3-70B', 'custom-Mixtral-8x7B-Instruct-v0p1', 'custom-Mistral-7B-Instruct-v0p3', 'custom-Qwen2-72B-Instruct', "deepseek-chat", 'custom-vicuna-13b-v1p5-16k'] 


                models +=  ['step2-ppo-2412-b2404-r2412-d2412-cstlr06-v0',
                    'step2-ppo-wenyu-v18p3-novel-it1860-1223',
                    'step1bm-20b-ppo-v17p3-merge3-622-1111']


                rag_settings = [{'actor_retrieval': False, 'retrieval_target': None}]#, {'actor_retrieval': True, 'retrieval_target': 'mixed'}]

                model_args = [(test_dataset_path, model, args, rag_setting, nth_exp) for model in models for rag_setting in rag_settings ]

                if 1:
                    # First run all generate tasks simultaneously
                    generate_futures = []
                    with ProcessPoolExecutor(max_workers=9) as generate_executor:
                        for model_arg in model_args:
                            future = generate_executor.submit(generate, model_arg)
                            generate_futures.append((future, model_arg))
                    
                    # As generate tasks complete, run up to 6 evaluate tasks at a time
                    with ProcessPoolExecutor(max_workers=6) as evaluate_executor:
                        evaluate_futures = []
                        
                        # Process completed generate tasks and submit evaluates
                        for generate_future, model_arg in generate_futures:
                            generate_future.result()  # Wait for generate to complete
                            evaluate_future = evaluate_executor.submit(evaluate, model_arg)
                            evaluate_futures.append(evaluate_future)
                        
                        # Process evaluate results as they complete
                        for future in evaluate_futures:
                            scores, results_by_case = future.result()
                            model_results.update(scores)
                            merge_global_results(global_results_by_case, results_by_case)

                else:
                    for model_arg in model_args:
                        generate(model_arg)
                        scores, results_by_case = evaluate(model_arg)
                        model_results.update(scores)
                        merge_global_results(global_results_by_case, results_by_case)
                    

                final_results[split] = model_results

            # save model results
            with open(test_dataset_path.replace('.json', f'_final_results.json'), 'w') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)
            
            if 'id' in final_results and 'ood' in final_results:
                # calculate avg
                avg_results = {}
                for model_args in model_results:
                    for dimension in model_results[model_args]:
                        avg_results.setdefault(model_args, {})
                        avg_results[model_args][dimension] = (final_results['id'][model_args][dimension] + final_results['ood'][model_args][dimension]) / 2

                final_results['avg'] = avg_results
            

            
            print(global_results_by_case)


            swapped_results = {}
            for actor_setting in global_results_by_case:
                for book_scene in global_results_by_case[actor_setting]:
                    if book_scene not in swapped_results:
                        swapped_results[book_scene] = {}
                    swapped_results[book_scene][actor_setting] = global_results_by_case[actor_setting][book_scene]
            global_results_by_case = swapped_results
            
            agent_settings = model_results.keys()
            win_rate = {}
            for agent_setting in agent_settings:
                win_rate[agent_setting] = {}
                for other_agent_setting in agent_settings:
                    if not any(model in agent_setting or model in other_agent_setting for model in ['gpt-4o', 'gpt3.5']):
                        continue
                    if agent_setting == other_agent_setting: 
                        win_rate[agent_setting][other_agent_setting] = 0.5
                        continue 
                    
                    try:
                        win_rate[agent_setting][other_agent_setting] = sum([1 for book_scene, model_results in global_results_by_case.items() if model_results[agent_setting]['score'] > model_results[other_agent_setting]['score']] + [0.5 for book_scene, model_results in global_results_by_case.items() if model_results[agent_setting]['score'] == model_results[other_agent_setting]['score']]) / max(1, len(global_results_by_case.items()))

                        print(f'Shared Secens between {agent_setting} {other_agent_setting} {len(global_results_by_case.items())} {win_rate[agent_setting][other_agent_setting]}')
                    except:
                        import pdb; pdb.set_trace()

                    


            logger.info(f'Win rate:\n{json.dumps(win_rate, ensure_ascii=False, indent=2)}')

            global_results_by_case = {k: sorted(v.items(), key=lambda x: x[1]['score'], reverse=True) for k, v in global_results_by_case.items()}
            
            print(f'Saving global eval results to {test_dataset_path.replace(".json", "nb={args.n_books}_continue={args.continue_from}_global_results_by_case.txt")}')

            with open(test_dataset_path.replace('.json', f'_global_results_by_case_0109.txt'), 'w') as f:
                f.write(f'Win rate:\n')
                for model in models:
                    f.write(f'{model}: {win_rate[model]}\n')
                f.write('\n')
                for book_scene, model_results in global_results_by_case.items():
                    f.write(f'{book_scene}\n')
                    for i_model, (model, results) in enumerate(model_results):
                        f.write(f'Ranked {i_model}: {model}\n')
                        for k, v in results.items():
                            if k == 'simulation_text':
                                continue
                            elif k == 'simulation':
                                f.write(f'==={k}===\n')
                                for _ in v:
                                    f.write(f"{_['content']}\n")
                            else:
                                if isinstance(v, dict): 
                                    f.write(f'==={k}===:\n{json.dumps(v, ensure_ascii=False, indent=2)}\n')
                                else:
                                    f.write(f'==={k}===:\n{v}\n')
                            f.write('\n')

            logger.info(f'Final results:\n{json.dumps(final_results, ensure_ascii=False, indent=2)}')
            logger.info(all_model_results_str)
            