clear

python data_construction/main.py --input data/src/books_example.jsonl --num_workers 1 --model Qwen2.5-7B-Instruct





# 从原始文件加成book
python ./tools/merge_xiaoshuo.py 


# Curate Data for Each Book
python data_construction/main.py --input data/src/fanren_merge.jsonl --num_workers 1 --model Qwen2.5-7B-Instruct

# Convert the Book Data into Training Samples & Test Set

python data_construction/transform.py

# Evaluation

python gca_evaluation/main.py --test_file data/test/test_set.json --actor_model gpt-4o --judge_model gpt-4o


