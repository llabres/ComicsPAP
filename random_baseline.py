from utils import parse_arguments, build_dataset
import json
from collections import defaultdict
from tqdm import tqdm

def compute_random_baseline(args):
    overall_results = {
        skill: {"num_examples": 0, "accuracy": 0} for skill in args.skill
    }
    
    # For each skill
    for skill in args.skill:
        print(f"Processing {skill}...")
        # Load dataset
        dataset = build_dataset(args.dataset, skill, args.split, args.single_image, args)
        
        # Count samples and compute theoretical random accuracy
        num_samples = len(dataset)
        options_count = defaultdict(int)
        
        # Count how many samples have each number of options
        for sample in tqdm(dataset):
            num_options = len(sample['options'])
            options_count[num_options] += 1
        
        # Compute weighted average accuracy
        theoretical_accuracy = sum(
            count * (1 / num_options)
            for num_options, count in options_count.items()
        ) / num_samples
        
        # Store results
        overall_results[skill]["num_examples"] = num_samples
        overall_results[skill]["accuracy"] = theoretical_accuracy
        overall_results[skill]["options_distribution"] = {
            str(k): v for k, v in options_count.items()
        }
        
        print(f"{skill}: {theoretical_accuracy:.4f} (random) with {num_samples} samples")

    # Calculate overall results
    total_examples = sum(data["num_examples"] for data in overall_results.values())
    overall_accuracy = sum(
        data["num_examples"] * data["accuracy"] 
        for data in overall_results.values()
    ) / total_examples

    # Save results
    overall_summary = {
        "overall_accuracy": overall_accuracy,
        "total_samples": total_examples,
        "skill_results": overall_results
    }

    output_path = f"save/random_baseline_{args.split}.json"
    with open(output_path, "w") as f:
        json.dump(overall_summary, f, indent=2)
    
    print(f"\nOverall theoretical random accuracy: {overall_accuracy:.4f}")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    args = parse_arguments()
    compute_random_baseline(args)
