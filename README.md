# CoSER

Official Code for "CoSER: Coordinating LLM-Based Persona Simulation of Established Roles"

Our data is currently undergoing internal review for safety concerns. The complete dataset and models will be released after safety checks, expected in February 2025. We have released the unorganized code to help understand our implementation. The organized version of the code will also be ready in February 2025.

## Setup

Install necessary dependencies via:

```bash
pip install -r requirements.txt
```

Setup your api_key and base_url for LLMs, in config.json. 

## Data 

The complete dataset is currently undergoing internal review for safety concerns. We have provided some example data from *The Harry Potter series* and *A Song of Ice and Fire series* in the example_data/ directory.

### Constructing Your Own Datasets

#### Prepare the Source Content of Interested Books (or Other Fictional Works)

To get started, you'll need to prepare a JSONL file containing the books you're interested in. Each line should contain a JSON object with the following structure:

```json
{"title":"Pride and Prejudice", "author": "Jane Austen", "content": "..."}
{"title":"The Picture of Dorian Gray", "author": "Oscar Wilde", "content": "..."}
{"title":"Emily Bronte", "author": "Wuthering Heights", "content": "..."}
```

Each JSON object should include three fields:
- `title`: The book's title
- `author`: The author's name
- `content`: The complete text content of the book

Alternatively, you can use our provided dataset [CoSER-Books-Gutenberg](https://huggingface.co/datasets/Neph0s/CoSER-Books-Gutenberg). This dataset is a subset of books used in the CoSER project. It contains 81 carefully selected classic books from Project Gutenberg. All books in this collection are in the public domain and freely accessible.

#### Curate Data for Each Book

To construct a CoSER-style dataset from your own books, run:

```bash
python data_construction/main.py --input data books_example.jsonl --num_workers 5
```

**Arguments**
- `--input`: Path to your input JSONL file containing the books data 
- `--output_dir`: Directory where the curated data will be saved (default: "data"). The final data for each book will be stored in data/final/ .
- `--num_workers`: Number of parallel workers for data processing (default: 1)
- `--model`: The LLM model to use for data construction (defaults to gpt-4o, though we employed claude-3-5-sonnet-20240620 when constructing CoSER dataset.)

**Note**: It is common to encounter parsing errors and other issues due to the inherent instability of LLMs when generating structured data. Our code includes comprehensive error handling and retry mechanisms to handle these cases gracefully. You can check the logs in `data_construction/main.log` for details about any errors and how they were processed.

#### Convert the Book Data into Training Samples & Test Set 

This step transforms the curated book data into: 1) training samples in sharegpt format, and 2) a test set. These data are used for given-circumstance acting evaluation (GCA) training and evaluation. 

```bash
python data_construction/transform.py 
```

**Arguments**
- `--dir`: Set as the output_dir in the previous step (default: data).

The script will generate:
- Training data: `data/train/sft_sharegpt.json`
- Test set: `data/test/test_set.json`

## Evaluation 

To evaluate an LLM' role-playing performance via Given-Circumtance Acting (GCA):

```bash
python gca_evaluation/main.py --test_file data/test/test_set.json --actor_model gpt-4o --judge_model gpt-4o
```

**Arguments**
- `--test_file`: Path to test dataset (default: data/test/test_set.json)
- `--actor_model`: Model used for character role-playing (default: gpt-4o)
- `--judge_model`: Model used for evaluation (default: gpt-4o)
- `--env_model`: Model for environment responses (default: gpt-4o)
- `--nsp_model`: Model for next-speaker prediction (default: gpt-4o-mini). For better cost-efficiency, we recommend using CoSER-70B or other self-deployed models
- `--retrieval`: Enable retrieval augmentation [None|raw_text|expr1|expr3|conv1|expr3_conv1|expr10_conv1]
- `--wo_thought`: Disable inner thoughts in GCA simulation

The evaluation process consists of two stages:
1. Simulation: Generated conversations are saved to exp/simulation/
2. Judging: Assessment results are saved to exp/evaluation/

The evaluation adopts two types of metrics:
1. LLM Judge Scoring (0-100) in terms of:
   - Storyline Consistency: Alignment with original dialogue
   - Anthropomorphism: Human-like behavior
   - Character Fidelity: Faithful character portrayal
   - Storyline Quality: Natural conversation development
   - Average Score (of the above dimensions)
2. Automated Metrics (comparing generated conversations with ground truth dialogues):
   - BLEU
   - ROUGE-L
