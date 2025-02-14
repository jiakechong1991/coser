# transforming data from results/final/... to the chat format
# such as : 
# [
#   [
#     {
#       "from": "Human",
#       "value": "Hello, how are you?"
#     },
#     {
#       "from": "Assistant",
#       "value": "I'm fine, thank you!"
#     }
#   ]
# ]

import os
import random


directory = 'results/full/'

# Get a list of all files in the directory
files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def stable_shuffle(files):
    # Python's hash() is randomized per process for security
    # To get consistent results, we'll use a simple string hash function
    def string_hash(s):
        h = 0
        for c in s:
            h = (31 * h + ord(c)) & 0xFFFFFFFF
        return h
    
    # Use our deterministic hash function instead of built-in hash()
    files_with_hash = [(f, string_hash(f)) for f in files]
    
    # Sort based on hash values
    files_with_hash.sort(key=lambda x: x[1])
    
    # Return just the filenames
    return [f[0] for f in files_with_hash]

# Shuffle the files in a stable way
files = stable_shuffle(files)

assert(files[:10] == ['The Amulet of Samarkand (Bartimaeus, #1).json', 'Choke.json', 'The Outsiders.json', 'The Witches.json', 'A Short History of Nearly Everything.json', 'Spirit Bound (Vampire Academy, #5).json', 'Alanna: The First Adventure (Song of the Lioness, #1).json', 'Lover Awakened (Black Dagger Brotherhood, #3).json', 'Twelfth Night.json', 'Boy of Chaotic Making (Whimbrel House, #3).json'])
print(files[:10])


count_remove = 0 

import argparse
from utils import remove_inner_thoughts

# Set up argument parser
parser = argparse.ArgumentParser(description='Transform book data to chat format')
args = parser.parse_args()

dedup = False

ENVIRONMENT = 'Environment'
NSP = 'NSP'

import json
import re

def transform_to_chat_data(file):
    with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    book_name = os.path.splitext(file)[0]

    character_datasets = data['character_datasets']
    character_profiles = {name: data['character_datasets'][name]['profile'] for name in character_datasets.keys()}

    import math 

    character_weight = {name: max(1, math.sqrt(len(data['character_datasets'][name]['plots']))) for name in character_datasets.keys()}

    characters = character_datasets.keys()
    
    train_chat_data = []
    test_chat_data = []
    
    n_plots = len(data['plots'])

    split_index = data.get('split_plot_index', int(n_plots * 0.9))

    count_random_next_actor = 0
    count_next_actor = 0

    message_keys = ['message', 'thought', 'description']

    for thoughtless in [True, False]:
        for with_other_character_profiles in [True, False]:
            for i_p, plot in enumerate(data['plots']):
                plot.pop('text', None)

                # check plot['key_characters']
                plot_key_characters = [] 
                for c in plot['key_characters']:
                    if 'name' in c:
                        plot_key_characters.append(c)
                    elif 'character' in c:
                        c['name'] = c.pop('character')
                        plot_key_characters.append(c)
                
                plot['key_characters'] = plot_key_characters

                for i_c, conversation in enumerate(plot['conversation']):
                    from collections import Counter
                    # Step 1: Count the number of each character's utterance
                    utterance_counts = Counter(utterance['character'] for utterance in conversation['dialogues'])
        
                    speaking_characters_w_env = sorted(utterance_counts, key=lambda x: utterance_counts[x], reverse=True) # 包括Environment

                    plot_characters = [ c['name'] for c in plot['key_characters']] 

                    # check conversation['key_characters']
                    conversation_key_characters = []
                    for c in conversation['key_characters']:
                        if 'name' in c:
                            conversation_key_characters.append(c)
                        elif 'character' in c:
                            c['name'] = c.pop('character')
                            conversation_key_characters.append(c)
                    
                    conversation['key_characters'] = conversation_key_characters

                    # check conversation['dialogues']
                    broken_sign = False

                    for i_u, utterance in enumerate(conversation['dialogues']):
                        if not 'message' in utterance:
                            if 'thought' in utterance:
                                utterance['message'] = utterance.pop('thought')
                            elif 'description' in utterance:
                                utterance['message'] = utterance.pop('description')
                            else:
                                # 看看utterance中有没有一个k, v 其中k以[开头 以]结尾
                                message = None
                                for k, v in utterance.items():
                                    if k.startswith('[') and k.endswith(']'):
                                        message = k + v
                                        break
                                if message is not None:
                                    utterance['message'] = message
                                    utterance.pop(k)
                                else:
                                    broken_sign = True
                                    import pdb; pdb.set_trace()
                                    break
                        
                        if not 'character' in utterance:
                            if 'name' in utterance:
                                utterance['character'] = utterance.pop('name')
                            else:
                                broken_sign = True
                                import pdb; pdb.set_trace()
                                break
                    
                    if broken_sign:
                        continue

                    if len([c for c in utterance_counts if c != ENVIRONMENT]) == 0:
                        # no valid characters in this conversation
                        continue

                    major_characters = [c for c in speaking_characters_w_env if c in character_profiles]
                    
                    if 'major_characters' not in conversation:
                        conversation['speaking_characters_w_env'] = speaking_characters_w_env
                        conversation['major_characters'] = major_characters
                    else:
                        assert conversation['speaking_characters_w_env'] == speaking_characters_w_env
                        assert conversation['major_characters'] == major_characters

                    if random.random() < 0.5:
                        scenario_str = conversation['scenario'] + '\n' + plot['summary']    
                    else:
                        scenario_str = conversation['scenario']

                    if dedup:
                        # Step 2: Select the character
                        if random.random() < 0.5:
                            # select the most frequent character
                            selected_character = speaking_characters_w_env[0]
                        else:
                            count = 0
                            while True:
                                selected_character = random.choice(conversation['dialogues'])['character']
                                if selected_character in speaking_characters_w_env:
                                    break
                                count += 1
                        
                        selected_characters = [selected_character]
                    else:
                        selected_characters = speaking_characters_w_env

                    # first collect character_profiles in this conversation 
                    tmp_character_profiles = {}
                    for character in speaking_characters_w_env:
                        if character == ENVIRONMENT:
                            continue
                        character_profile = ''

                        if character in character_profiles:
                            character_profile = character_profiles[character]
                        
                        if character in plot_characters:
                            character_info = [c for c in plot['key_characters'] if c.get('name', '') == character ][0]

                            if 'description' in character_info:
                                character_profile = character_info.get('description', '').strip('\n') + '\n\n' + character_profile.strip('\n')
                        
                        character_profile = character_profile.strip(' \n')
                        if character_profile != '':
                            tmp_character_profiles[character] = character_profile

                    for character in selected_characters:
                        chat = []
                        
                        if character == ENVIRONMENT:
                            from utils import get_environment_prompt
                            system_prompt = get_environment_prompt(major_characters=major_characters,  scenario=scenario_str)
                        else:
                            # get the 'thought' 
                            character_info = [char for char in conversation['key_characters'] if char.get('name', '') ]

                            if len(character_info) == 0: 
                                thought = ""
                            else:
                                try:
                                    thought = character_info[0]['thought']
                                except:
                                    thought = character_info[0].get('description', '')

                                # drop out the thought at a chance of 0.5
                                if random.random() < 0.5:
                                    thought = ""

                            # get the 'system_prompt'
                            character_profile = tmp_character_profiles.get(character, '')

                            if character in plot_characters:
                                character_info = [c for c in plot['key_characters'] if c.get('name', '') == character ][0]

                                if random.random() < 0.5:
                                    character_profile += '\n\n' + character_info.get('summary', '')
                            
                            character_profile = character_profile.strip(' \n')

                            #print(f'{character}: {character_profile}')
                            
                            from utils import get_character_prompt
                            system_prompt = get_character_prompt(book_name, character, character_profile, plot, conversation, thought, thoughtless, other_character_profiles=tmp_character_profiles if with_other_character_profiles else None)

                        
                        chat.append({"from": "system", "value": system_prompt})



                        chat.append({"from": "user", "value": "===Conversation Start===\n\n"})
                        
                        prev_role = 'human'

                        import re

                        for i_u, utterance in enumerate(conversation['dialogues']):

                            message = utterance['message']

                            if utterance['character'] != character:
                                if utterance['character'] != ENVIRONMENT:
                                    message = remove_inner_thoughts(message)

                                if prev_role == 'human':
                                    chat[-1]['value'] += f"{utterance['character']}: {message}\n\n"
                                else:
                                    chat.append({"from": "user", "value": f"{utterance['character']}: {message}\n\n"})
                                prev_role = 'human'
                            else:
                                if thoughtless:
                                    if utterance['character'] != ENVIRONMENT:
                                        message = remove_inner_thoughts(message)

                                    if len(message.strip(' \n')) == 0:
                                        # remove training data that may contain empty outputs
                                        continue
                                    
                                if prev_role == 'character':
                                    chat[-1]['value'] += f"{message}\n\n"
                                else:
                                    chat.append({"from": "assistant", "value": f"{utterance['character']}: {message}\n\n"})
                                prev_role = 'character'
                        
                        if len([m for m in chat if m['from'] == 'assistant']) == 0:
                            continue 
                            
                
                        chat = {
                            "conversations": chat,
                            "details": {
                                "book_name": book_name,
                                "plot_index": i_p,
                                "plot": plot,
                                "character": character
                            }
                        }
                        
                        if i_p < split_index:
                            # split train/test data by plots. the first 80% of the plots are used for training, and the last 20% are used for testing.

                            train_chat_data.append(chat)
                        else:
                            if thoughtless == False and with_other_character_profiles == True:  
                                test_chat_data.append(chat)
                    
                    # add the next speaker prediction data
                    from utils import get_nsp_prompt
                    chat = []
                    system_prompt = get_nsp_prompt(all_characters=speaking_characters_w_env, scenario=scenario_str)

                    chat.append({"from": "system", "value": system_prompt})
                    chat.append({"from": "user", "value": "===Conversation Start===\n\n"})

                    for i_u, utterance in enumerate(conversation['dialogues']):
                        next_actor = utterance['character']
                                        
                        # drop out the next actor in some random cases
                        if next_actor not in major_characters + [ENVIRONMENT]:
                            if random.random() < 0.25:
                                next_actor = "random"
                        elif next_actor not in speaking_characters_w_env:
                            # almost impossible
                            next_actor = "random"
                                        
                        if next_actor == "random": count_random_next_actor += 1
                        count_next_actor += 1

                        chat.append({"from": "assistant", "value": next_actor})

                        message = None
                        for key in message_keys:
                            if key in utterance:
                                message = utterance[key]
                                break
                        
                        if message is None:
                            break

                        chat.append({"from": "user", "value": f"{utterance['character']}: {message}\n\n"})
                        
                    
                    chat.append({"from": "assistant", "value": "<END CHAT>"})

                    chat = {"conversations": chat, "details": {
                        "book_name": book_name,
                        "plot_index": i_p,
                        "plot": plot,
                        "character": NSP
                    }}

                    if i_p < split_index and random.random() < 0.1:
                        train_chat_data.append(chat)



    # plot-level dataset
    for i_p, plot in enumerate(data['plots']):
        plot['i_p'] = i_p

    def avg(lst):
        return sum(lst) / len(lst)

    test_plots = data['plots'][split_index:]
    # calculate the weight of each plot as the sum of the weights of the key characters in this plot
    for plot in test_plots:
        for conversation in plot['conversation']:
            try:
                conversation['weight'] = avg([character_weight.get(c, 1) for c in conversation['speaking_characters_w_env'] if c != ENVIRONMENT])
            except:
                conversation['weight'] = 1

    return train_chat_data, test_chat_data, test_plots, character_profiles

train_chat_data = []
test_chat_data_id_books = {}
test_chat_data_ood_books = {}
test_chat_data_got = {}

# Process each file
n_b = len(files)

test_plots_id_books = {}
test_plots_ood_books = {}
test_plots_got = {}


for i_b, file in enumerate(files):
    book_name = os.path.splitext(file)[0]
    train_chat_data_book, test_chat_data_book, test_plots_book, character_profiles_book = transform_to_chat_data(file)


    if i_b < n_b * 0.9 and not ('A Song of Ice and Fire' in book_name):
        # The training set excludes the Got data 
        train_chat_data += train_chat_data_book
        test_chat_data_id_books[book_name] = test_chat_data_book
        test_plots_id_books[book_name] = {
            "character_profiles": character_profiles_book,
            "plots": test_plots_book,
        }
    else:
        if 'A Song of Ice and Fire' in book_name:
            test_chat_data_got[book_name] = test_chat_data_book
            test_plots_got[book_name] = {
                "character_profiles": character_profiles_book,
                "plots": test_plots_book,
            }

        test_chat_data_ood_books[book_name] = test_chat_data_book # train_chat_data_book + test_chat_data_book
        test_plots_ood_books[book_name] = {
            "character_profiles": character_profiles_book,
            "plots": test_plots_book,
        }
    

# print key statistics of the datasets
print(f"n_b: {n_b}")
print(f"count_remove: {count_remove}")
print(f"train_chat_data: {len(train_chat_data)}")
print(f"test_chat_data_id_books: {len(test_chat_data_id_books)}")
print(f"test_chat_data_ood_books: {len(test_chat_data_ood_books)}")

assert(set(test_chat_data_ood_books.keys()) == set(['A Dance with Dragons (A Song of Ice and Fire, #5)', 'A Game of Thrones (A Song of Ice and Fire, #1)', 'A Clash of Kings  (A Song of Ice and Fire, #2)', 'A Feast for Crows (A Song of Ice and Fire, #4)', 'A Storm of Swords (A Song of Ice and Fire, #3)', 'A Discovery of Witches (All Souls, #1)', 'Tess of the D’Urbervilles', 'The Road', 'The Last Song', 'A Scanner Darkly', 'James and the Giant Peach', 'Before I Fall', 'The Forgotten Garden', 'The Son of Neptune (The Heroes of Olympus, #2)', 'Foundation (Foundation, #1)', 'The Summer I Turned Pretty (Summer, #1)', 'The Hunger Games (The Hunger Games, #1)', 'Frankenstein: The 1818 Text', 'Tuck Everlasting', 'Halfway to the Grave (Night Huntress, #1)', 'On the Beach', 'Leaves of Grass', 'The Girl with the Dragon Tattoo (Millennium, #1)', 'A Story of Yesterday', "I'll Give You the Sun", 'The Dark Tower (The Dark Tower, #7)', 'The Amber Spyglass (His Dark Materials, #3)', 'Percy Jackson and the Olympians (Percy Jackson and the Olympians, #1-3)', 'The Glass Menagerie', 'Little Women', 'The Name of the Rose', 'To the Lighthouse', 'Les Misérables', 'Anna Karenina', 'The Hunt for Red October (Jack Ryan, #3)', 'The V Girl: A Coming of Age Story', "The Pilgrim's Progress", "Always Remember: Ben's Story (Ravenswood, #3)", 'The Murder of Roger Ackroyd (Hercule Poirot, #4)', "To All the Boys I've Loved Before (To All the Boys I've Loved Before, #1)", 'The Woman in White', 'The Fiery Cross (Outlander, #5)', 'The Man in the High Castle', "At Grave's End (Night Huntress, #3)", 'Twenty Love Poems and a Song of Despair', 'The Girl on the Train', 'Fear and Loathing in Las Vegas', 'Where She Went (If I Stay, #2)', 'The Little Prince', 'Something Wicked This Way Comes', 'A Light in the Attic', 'The Secret History', 'Tales of H P  Lovecraft', 'Notes from Underground', 'Naked Lunch', 'Out of My Mind (The Out of My Mind Series)', 'Wuthering Heights', 'Pretties (Uglies, #2)', 'The Song of Achilles', 'The Tales of Beedle the Bard (Hogwarts Library, #3)', 'Seabiscuit: An American Legend', 'Dubliners', 'Dear John', 'Eleanor & Park', 'Jaws (Jaws, #1)', "The Restaurant at the End of the Universe (The Hitchhiker's Guide to the Galaxy, #2)", 'The Call of the Wild', 'Corelli’s Mandolin', 'Hard-Boiled Wonderland and the End of the World', 'The Complete Novels', 'The Taming of the Shrew', 'The Guernsey Literary and Potato Peel Pie Society', 'Tender Is the Night', 'Storm Front (The Dresden Files, #1)', 'Number the Stars', 'Cold Mountain', 'My Name Is Asher Lev', 'I, Robot (Robot, #0 1)', 'Water for Elephants', 'The Forgotten Palace', "A Dog's Purpose (A Dog's Purpose, #1)", 'The Girl Who Kicked the Hornet’s Nest (Millennium, #3)']))

random.shuffle(train_chat_data)

folder_path = 'rp' 
if not folder_path:
    folder_path = 'vanilla'

folder_path = 'results/final/' + folder_path + '/'

# mkdir if not exists
os.makedirs(folder_path, exist_ok=True)


# Save the chat data
## sharegpt format
def to_sharegpt_format(chat_data):
    results = []
    role_map = {
        'system': 'system',
        'user': 'human',
        'assistant': 'assistant',
    }
    for sample in chat_data:
        conversation = []
        for message in sample['conversations']:
            conversation.append({
                "from": role_map[message['from']],
                "value": message['value']
            })
        results.append({"conversations": conversation})
    return results

with open(folder_path + 'train_conversations_sharegpt.json', 'w', encoding='utf-8') as f:
    sharegpt_format_data = to_sharegpt_format(train_chat_data)
    json.dump(sharegpt_format_data, f, ensure_ascii=False, indent=2)

## step1 format
def to_step1_format(chat_data):
    results = []
    role_map = {
        'system': 'System',
        'user': 'Human',
        'assistant': 'Assistant',
    }
    for sample in chat_data:
        conversation = []
        for message in sample['conversations']:
            conversation.append({
                "from": role_map[message['from']],
                "value": message['value']
            })
        results.append(conversation)
    return results

print(f'Saving into {folder_path}')

with open(folder_path + 'train_conversations_step1.json', 'w', encoding='utf-8') as f:
    step1_format_data = to_step1_format(train_chat_data)
    json.dump(step1_format_data, f, ensure_ascii=False, indent=2)

with open(folder_path + 'test_conversations_id_books.json', 'w', encoding='utf-8') as f:
    json.dump(test_chat_data_id_books, f, ensure_ascii=False, indent=2)

with open(folder_path + 'test_conversations_ood_books.json', 'w', encoding='utf-8') as f:
    json.dump(test_chat_data_ood_books, f, ensure_ascii=False, indent=2)

with open(folder_path + 'test_conversations_got.json', 'w', encoding='utf-8') as f:
    json.dump(test_chat_data_got, f, ensure_ascii=False, indent=2)

import copy
def to_test_circumstance(test_plots, n_samples=100, output_path=None):
    samples = []
    for book_name in test_plots.keys():
        for i, plot in enumerate(test_plots[book_name]['plots']):
            for j, conversation in enumerate(plot['conversation']):
                weight = conversation.get('weight', 1.0)  # Default weight of 1.0 if not specified
                samples.append((book_name, i, j, weight))
    
    weights = [s[-1] for s in samples]
    print('original average weight:', sum(weights) / len(weights))

    # first select the top half of the samples
    samples = sorted(samples, key=lambda x: x[-1], reverse=True)
    samples = samples[:len(samples) // 2]

    weights = [s[-1] for s in samples]
    print('after first selection average weight:', sum(weights) / len(weights))

    import pdb; pdb.set_trace()

    import random
    random.seed(42)

    import numpy as np
    np.random.seed(42)

    # Sample n_samples items according to weights
    selected_samples = []
    book_counts = {}

    weights = np.array(weights)
    weights = weights / weights.sum() # Normalize weights to probabilities
    
    while len(selected_samples) < n_samples:
        # Sample one index according to weights
        idx = np.random.choice(len(weights), p=weights)
        sample = samples[idx]
        book_name = sample[0]
        
        # Check if we can add this sample
        book_counts.setdefault(book_name, 0) 

        if sample in selected_samples or book_counts[book_name] >= 5:
            continue
        
        book_counts[book_name] += 1
        selected_samples.append(sample)
    
    samples = selected_samples

    assert(len(set(samples)) == len(samples))


    print('after sampling average weight:', sum([s[-1] for s in samples]) / len(samples))

    sampled_conversations = []
    # now arrange the data into a list 
    for book_name, i, j, weight in samples:
        plot = copy.deepcopy(test_plots[book_name]['plots'][i])
        conversation = copy.deepcopy(plot['conversation'][j])
        plot.pop('conversation', None)
        
        for _ in conversation['key_characters']:
            _.pop('i_p', None)
            _.pop('i_c', None)
        
        for _ in conversation['dialogues']:
            _.pop('i_p', None)
            _.pop('i_c', None)
            _.pop('i_u', None)
        
        for _ in plot['key_characters']:
            _.pop('i_p', None)

        conversation['plot'] = plot
        conversation['character_profiles'] = { c: test_plots[book_name]['character_profiles'][c] for c in conversation['major_characters'] }
        conversation['book'] = book_name
        conversation['i_c'] = j

        sampled_conversations.append(conversation)
        print(f'Test Circumstance {book_name} {i} {j}')

    return sampled_conversations

# Save the test plots
with open(folder_path + 'test_plots_id_books.json', 'w', encoding='utf-8') as f:
    json.dump(test_plots_id_books, f, ensure_ascii=False, indent=2)

with open(folder_path + 'test_circumstance_id.json', 'w', encoding='utf-8') as f:
    test_circumstance_id = to_test_circumstance(test_plots_id_books, n_samples=100)
    json.dump(test_circumstance_id, f, ensure_ascii=False, indent=2)

with open(folder_path + 'test_plots_ood_books.json', 'w', encoding='utf-8') as f:
    json.dump(test_plots_ood_books, f, ensure_ascii=False, indent=2)

with open(folder_path + 'test_circumstance_ood.json', 'w', encoding='utf-8') as f:
    test_circumstance_ood = to_test_circumstance(test_plots_ood_books, n_samples=100)
    json.dump(test_circumstance_ood, f, ensure_ascii=False, indent=2)

# collect the interested books and characters in test_circumstance_id and test_circumstance_ood
with open(folder_path + 'interested_books_characters.json', 'w', encoding='utf-8') as f:
    interested_books_characters = {}
    for conversation in test_circumstance_id + test_circumstance_ood:
        book_title = conversation['book']
        interested_books_characters.setdefault(book_title, {})
        interested_books_characters[book_title].update(conversation['character_profiles'])

    num_characters = sum([len(v) for v in interested_books_characters.values()])
    print(f'Num Books {len(interested_books_characters)}, Num Characters {num_characters} ')
    json.dump(interested_books_characters, f, ensure_ascii=False, indent=2)

with open(folder_path + 'test_plots_got.json', 'w', encoding='utf-8') as f:
    json.dump(test_plots_got, f, ensure_ascii=False, indent=2)
