import re
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import parse_arguments, start_experiment, save_results, build_model, build_dataset
import os
import json

# CUDA_VISIBLE_DEVICES=0 python evaluate.py --split test --skill sequence_filling char_coherence visual_closure text_closure caption_relevance --model ../models/Qwen2.5-VL-3B-Instruct --eval_batch_size 10 --dataset_cache /data/users/evivoli/datasets/dcm_22k/stories/anns/single_image --single_image


def extract_prediction(text):
    """Extract prediction number from model output."""
    # Try to find "answer: X" pattern
    answer_match = re.search(r'answer:\s*(\d+)', text.lower())
    if answer_match:
        return int(answer_match.group(1))
    
    # If no "answer: X" pattern, try to find any number
    number_match = re.search(r'\d+', text)
    if number_match:
        return int(number_match.group(0))
    
    # Default to -1 if no number found
    return -1

def evaluate_skill(model, dataset, collator, device, skill, args):
    model.eval()
    all_outputs = []
    all_metadata = []

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=20,
        collate_fn=collator
    )

    # First pass: collect all model outputs
    for inputs, metadata in tqdm(dataloader, desc=f"Evaluating {skill}"):
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
            )
        
        all_outputs.extend(outputs.cpu())
        all_metadata.append(metadata)

    
    # Batch decode all outputs
    print(f"Decoding {skill} outputs...")
    predictions = collator.processor.batch_decode(all_outputs, skip_special_tokens=True)
    
    # Process all predictions at once
    print(f"Processing {skill} predictions...")
    numerical_preds = [extract_prediction(pred) for pred in predictions]
    
    # Combine metadata
    combined_metadata = {
        'sample_ids': sum((m['sample_ids'] for m in all_metadata), []),
        'labels': sum((m['labels'] for m in all_metadata), [])
    }
    
    # Create results
    results = []
    if args.split == 'val':
        total_correct = 0
        total_samples = 0
    
    for pred_text, pred_num, sample_id, solution_idx in zip(
        predictions, numerical_preds, 
        combined_metadata['sample_ids'], 
        combined_metadata['labels']
    ):
        result = {
            'sample_id': sample_id,
            'full_prediction': pred_text,
            'prediction': pred_num,
            'skill': skill,
        }
        
        if args.split == 'val':
            result['solution_index'] = solution_idx
            correct = pred_num == solution_idx
            result['correct'] = correct
            total_correct += correct
            total_samples += 1
        
        results.append(result)
    
    # Add summary result for validation set
    if args.split == 'val':
        summary = {
            'sample_id': 'summary',
            'accuracy': total_correct / total_samples,
            'total_samples': total_samples,
        }
        results.insert(0, summary)
    
    return results

def main():
    args = parse_arguments()
    experiment_name = start_experiment(args)
    
    # Build model and move to device
    with torch.no_grad():
        model, collator = build_model(args)

    device = next(model.parameters()).device
    
    # Track overall results
    overall_results = {
        skill: {"num_examples": 0, "correct": 0} for skill in args.skill
    }
    
    # Evaluate each skill
    for skill in args.skill:
        # Load dataset
        dataset = build_dataset(args.dataset, skill, args.split, args.single_image, args)
        
        # Evaluate
        results = evaluate_skill(model, dataset, collator, device, skill, args)
        
        # Save results
        save_results(results, experiment_name, skill, args)
        
        # Update overall results for validation
        if args.split == 'val':
            summary = results[0]  # First item contains the summary
            overall_results[skill]["num_examples"] = summary["total_samples"]
            overall_results[skill]["accuracy"] = summary["accuracy"]
            
            # Log to wandb if enabled
            if args.wandb:
                import wandb as wb
                wb.log({
                    f"{skill}/accuracy": summary["accuracy"],
                    f"{skill}/samples": summary["total_samples"]
                })
    
    # Calculate and log overall results for validation
    if args.split == 'val':
        total_examples = sum(data["num_examples"] for data in overall_results.values())
        overall_accuracy = sum(
            data["num_examples"] * data["accuracy"] 
            for data in overall_results.values()
        ) / total_examples
        
        # Save overall results
        overall_summary = {
            "overall_accuracy": overall_accuracy,
            "total_samples": total_examples,
            "skill_results": overall_results
        }
        
        with open(os.path.join(args.save_dir, experiment_name, "overall_results.json"), "w") as f:
            json.dump(overall_summary, f, indent=2)
        
        # Log overall results to wandb
        if args.wandb:
            wb.log({
                "overall/accuracy": overall_accuracy,
                "overall/samples": total_examples
            })
            
            # Create a summary table
            if wb.run is not None:
                data = [[skill, data["accuracy"], data["num_examples"]] 
                       for skill, data in overall_results.items()]
                data.append(["Overall", overall_accuracy, total_examples])
                
                table = wb.Table(data=data, columns=["Skill", "Accuracy", "Samples"])
                wb.log({"results_table": table})

if __name__ == "__main__":
    main()
