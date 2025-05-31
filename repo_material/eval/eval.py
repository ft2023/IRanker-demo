import argparse
import importlib.util
import os
import sys

def load_module_from_file(file_path):
    """Load a Python module from a file path."""
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(description='Evaluation script for different tasks')
    
    # Task selection argument
    parser.add_argument('--task', type=str, required=True,
                      choices=['passage', 'rec', 'router'],
                      help='Task to evaluate (passage, rec, or router)')
    
    # Dataset argument
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset to evaluate on')
    
    # GPU ID argument
    parser.add_argument('--gpu_id', type=str, default='0',
                      help='GPU ID to use (default: 0)')
    
    # Model path argument
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model checkpoint to use for evaluation')
    
    args = parser.parse_args()
    
    # Map task to script file
    script_map = {
        'passage': 'eval_passage.py',
        'rec': 'eval_rec.py',
        'router': 'eval_router.py'
    }
    
    # Validate dataset based on task
    if args.task == 'passage':
        valid_datasets = ['passage_5', 'passage_7', 'passage_9']
        if args.dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset for passage task. Must be one of {valid_datasets}")
    elif args.task == 'rec':
        valid_datasets = ['movie', 'game', 'music']
        if args.dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset for rec task. Must be one of {valid_datasets}")
    elif args.task == 'router':
        valid_datasets = ['balance', 'cost', 'performance']
        if args.dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset for router task. Must be one of {valid_datasets}")
    
    # Get the script path
    script_path = os.path.join(os.path.dirname(__file__), script_map[args.task])
    
    # Load and run the appropriate module
    module = load_module_from_file(script_path)
    module.main(args.dataset, args.gpu_id, args.model_path)

if __name__ == "__main__":
    main() 