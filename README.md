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

To construct data for your own books of interest, run

```bash
cd data_construction
python process.py
```

### Evaluation via GCA (Given-Circumtance Acting)

```bash
python gca_evaluation/eval_reproduce.py
```



 
