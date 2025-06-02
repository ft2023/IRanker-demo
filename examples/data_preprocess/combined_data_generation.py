import re
import os
from datasets import Dataset, load_dataset, concatenate_datasets
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse


def gen_dataset(
        num_samples: int,
        num_operands: int = 6,
        max_target: int = 1000,
        min_number: int = 1,
        max_number: int = 100,
        operations: List[str] = ['+', '-', '*', '/'],
        seed_value: int = 42,
) -> List[Tuple]:
    """Generate dataset for countdown task.

    Args:
        num_samples: Number of samples to generate
        num_operands: Number of numbers provided in each sample
        max_target: Maximum value for target number
        min_number: Minimum value for provided numbers
        max_number: Maximum value for provided numbers
        operations: List of allowed operations
        seed_value: Random seed for reproducibility

    Returns:
        List of tuples containing (target, numbers, solution)
    """
    seed(seed_value)
    samples = []

    for _ in tqdm(range(num_samples)):
        # Generate random target
        target = randint(1, max_target)

        # Generate random numbers
        numbers = [randint(min_number, max_number) for _ in range(num_operands)]

        samples.append((target, numbers))

    return samples


def make_prefix(dp, template_type):
    """Generate prefix for the prompt based on template type."""
    query = dp['problem']
    prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n {query} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/rec')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--num_operands', type=int, default=6)
    parser.add_argument('--max_target', type=int, default=1000)
    parser.add_argument('--min_number', type=int, default=1)
    parser.add_argument('--max_number', type=int, default=100)
    parser.add_argument('--train_size', type=int, default=6000)
    parser.add_argument('--test_size', type=int, default=39)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')

    args = parser.parse_args()

    data_source = 'combined_ranking'
    base_path = '/data/taofeng2/tiny_rec/rank_dataset'

    # Define file paths for different data types
    file_mapping = {
        "train": {
            "Rec-Movie": f"{base_path}/data_rec/data/movie_cases_train.json",
            "Rec-Music": f"{base_path}/data_rec/data/music_cases_train.json",
            "Rec-Game": f"{base_path}/data_rec/data/game_cases_train.json",
            "Router-Performance": f"{base_path}/data_router/data_split/router_cases_all_Performance First_train.json",
            "Router-Balance": f"{base_path}/data_router/data_split/router_cases_all_Balance_train.json",
            "Router-Cost": f"{base_path}/data_router/data_split/router_cases_all_Cost First_train.json",
            "Passage-5": f"{base_path}/data_ms_marco/data/passage_cases_train_5_candidate.json",
            "Passage-7": f"{base_path}/data_ms_marco/data/passage_cases_train_7_candidate.json",
            "Passage-9": f"{base_path}/data_ms_marco/data/passage_cases_train_9_candidate.json"
        },
        "test": {
            "Rec-Movie": f"{base_path}/data_rec/data/movie_cases_test.json",
            "Rec-Music": f"{base_path}/data_rec/data/music_cases_test.json",
            "Rec-Game": f"{base_path}/data_rec/data/game_cases_test.json",
            "Router-Performance": f"{base_path}/data_router/data_split/router_cases_all_Performance First_test.json",
            "Router-Balance": f"{base_path}/data_router/data_split/router_cases_all_Balance_test.json",
            "Router-Cost": f"{base_path}/data_router/data_split/router_cases_all_Cost First_test.json",
            "Passage-5": f"{base_path}/data_ms_marco/data/passage_cases_test_5_candidate.json",
            "Passage-7": f"{base_path}/data_ms_marco/data/passage_cases_test_7_candidate.json",
            "Passage-9": f"{base_path}/data_ms_marco/data/passage_cases_test_9_candidate.json"
        }
    }

    # Load and combine training datasets
    datasets_list_train = []
    for file_path in file_mapping["train"].values():
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        dataset = load_dataset('json', data_files=file_path, split='train')
        print(f"Loaded dataset from {file_path}: {len(dataset)} examples")
        datasets_list_train.append(dataset)

    combined_dataset_train = concatenate_datasets(datasets_list_train).shuffle(seed=42)

    # Load and combine test datasets
    datasets_list_test = []
    for file_path in file_mapping["test"].values():
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        dataset = load_dataset('json', data_files=file_path, split='train')
        print(f"Loaded dataset from {file_path}: {len(dataset)} examples")
        datasets_list_test.append(dataset)

    combined_dataset_test = concatenate_datasets(datasets_list_test).shuffle(seed=42)
    combined_dataset_test = combined_dataset_test.select(range(min(200, len(combined_dataset_test))))

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            
            # Process different data types
            if example.get('candidate_passages') is not None:
                passage_dict = example['candidate_passages']
                candidate_passages = [key for key, value in passage_dict.items() if value is not None]
                candidate_ = candidate_passages
                gt = example['gt_passage']
            elif example.get('gt_item') is not None:
                candidate_ = example['candidate_items']
                gt = example['gt_item']
            elif example.get('ground_truth') is not None:
                candidate_ = example['candidates']
                gt = example['ground_truth']
            else:
                candidate_ = example['candidate_text']
                gt = example['gt_llm']

            solution = {
                'candidate_text': candidate_,
                'gt': gt
            }
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "ranking",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    # Process datasets
    train_dataset = combined_dataset_train.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = combined_dataset_test.map(function=make_map_fn('test'), with_indices=True)

    # Save datasets
    os.makedirs('data/ranking_combine', exist_ok=True)
    train_dataset.to_parquet(os.path.join('data/ranking_combine', 'train.parquet'))
    test_dataset.to_parquet(os.path.join('data/ranking_combine', 'test.parquet'))

    # Handle HDFS if specified
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)