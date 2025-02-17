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
{"title":"Pride and Prejudice", "author": "Jane Austen", "content": ...}
{"title":"The Picture of Dorian Gray", "author": "Oscar Wilde", "content": ...}
{"title":"Emily Bronte", "author": "Wuthering Heights", "content": ...}
```

Each JSON object should include three fields:
- `title`: The book's title
- `author`: The author's name
- `content`: The complete text content of the book

Alternatively, you can use our provided dataset [CoSER-Books-Gutenberg](https://huggingface.co/datasets/Neph0s/CoSER-Books-Gutenberg). This dataset is a subset of books used in the CoSER project. It contains 81 carefully selected classic books from Project Gutenberg. All books in this collection are in the public domain and freely accessible.

#### Run the Construction Code

To construct a CoSER-style dataset from your own books, run:

```bash
python data_construction/process.py --input data/src/books_example.jsonl --output_dir data/curated/ --num_workers 5
```

### Arguments
- `--input`: Path to your input JSONL file containing the books data 
- `--output_dir`: Directory where the curated data will be saved (default: "data/curated/")
- `--num_workers`: Number of parallel workers for data processing (default: 1)


### Evaluation via GCA (Given-Circumtance Acting)

```bash
python gca_evaluation/eval_reproduce.py
```



 
