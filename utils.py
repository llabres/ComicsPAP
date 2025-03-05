import os
import json
import argparse
import torch
from datetime import datetime
from datasets import load_dataset, load_from_disk

def parse_arguments():
    parser = argparse.ArgumentParser(description="Pick A Panel Dataset Training and Evaluation")
    parser.add_argument("--dataset", type=str, default="VLR-CVC/ComicsPAP", help="Dataset HuggingFace repo name or path")
    parser.add_argument("--split", type=str, default="val", help="Dataset split")
    parser.add_argument("--skill", type=str, nargs='+', help="Task name")
    parser.add_argument("--model", type=str, help="Model HuggingFace repo name or path")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Eval Batch size")
    parser.add_argument("--single_image", action="store_true", help="Whether to render the panels as a single image")
    parser.add_argument("--save_dir", type=str, default="save", help="Directory to save the results and checkpoints")
    parser.add_argument("--dataset_cache", type=str, default="dataset_cache", help="Directory to load/save the dataset cache. This is used to store the single_image dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb", action="store_true", help="Whether to log the results to Weights & Biases")
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment")
    
    # Training-specific arguments
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=0, help="Maximum number of training steps")
    parser.add_argument("--eval_steps", type=int, default=99999999999, help="Number of steps between evaluations")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps for gradient accumulation")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether to use gradient checkpointing")
    parser.add_argument("--lora", action="store_true", help="Whether to use LoRA")
    
    args = parser.parse_args()

    if type(args.skill) is str:
        args.skill = [args.skill]

    return args

def start_experiment(args):
    if args.experiment_name is None:
        args.experiment_name = f"{args.model.split('/')[-1]}{'_sft' if args.max_steps > 0 else ''}{'_lora' if args.lora else ''}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(os.path.join(args.save_dir, args.experiment_name), exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    if args.wandb:
        import wandb as wb
        wb.init(project="Pick-A-Panel", name=args.experiment_name, tags=[args.model.split("/")[-1]] + args.skill)
        wb.config.update(args)

    return args.experiment_name
        
def save_results(results, experiment_name, skill, args):
    with open(os.path.join(args.save_dir, experiment_name, f"results_{skill}.json"), "w") as f:
        json.dump(results, f, indent=2)

def build_model(args):
    if "qwen" in args.model.lower():
        from model_builders import build_qwen
        return build_qwen(args)
    elif "smol" in args.model.lower() or "idefics" in args.model.lower():
        from model_builders import build_smolvlm
        return build_smolvlm(args)
    elif "llama" in args.model.lower():
        from model_builders import build_llama
        return build_llama(args)
    else:
        raise ValueError("Model not supported")
    
def build_dataset(dataset_name_or_path, skill, split, single_image, args):
    if single_image:
        single_image_path = os.path.join(args.dataset_cache, f"{skill}_{split}_single_image")
        if os.path.exists(single_image_path):
            dataset = load_from_disk(single_image_path)
        else:
            from data_utils import SingleImagePickAPanel
            processor = SingleImagePickAPanel(font_path="save/Arial.ttf")
            dataset = load_dataset(dataset_name_or_path, skill, split=split)
            dataset = dataset.map(
                processor.map_to_single_image,
                batched=True,
                batch_size=32,
                remove_columns=['context', 'options']
            )
            
            dataset.save_to_disk(single_image_path)
    else:
        dataset = load_dataset(dataset_name_or_path, skill, split=split)

    return dataset




