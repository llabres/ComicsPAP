
import os
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
from datasets import interleave_datasets
from utils import parse_arguments, start_experiment, build_model, build_dataset

# CUDA_VISIBLE_DEVICES=3 python sft.py --skill sequence_filling char_coherence visual_closure text_closure caption_relevance --model ../models/SmolVLM-500M-Instruct --batch_size 8 --gradient_accumulation_steps 16 --max_steps 500 --eval_steps 250 --dataset_cache /data/users/evivoli/datasets/dcm_22k/stories/anns/single_image --single_image --gradient_checkpointing --wandb



if __name__ == "__main__":
    args = parse_arguments()

    experiment_name = start_experiment(args)

    train_datasets = []
    for skill in args.skill:
        dataset = build_dataset(args.dataset, skill, 'train', args.single_image, args)
        train_datasets.append(dataset)
    
    if len(train_datasets) > 1:
        # Calculate probabilities based on dataset sizes
        total_samples = sum(len(ds) for ds in train_datasets)
        probabilities = [len(ds)/total_samples for ds in train_datasets]
        train_dataset = interleave_datasets(
            train_datasets, 
            probabilities=probabilities,
            seed=args.seed
        )
    else:
        train_dataset = train_datasets[0]

    model, eval_collator, train_collator = build_model(args)

    if args.lora:
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

training_args = SFTConfig(
    output_dir=os.path.join(args.save_dir, experiment_name),
    max_steps=args.max_steps,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=True if args.gradient_checkpointing else False,
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",
    learning_rate=args.learning_rate,
    lr_scheduler_type="constant",
    # Logging and evaluation
    logging_steps=10,
    eval_steps=args.eval_steps,
    eval_strategy="no",
    save_strategy="steps",
    save_steps=args.eval_steps,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=False,
    # Mixed precision and gradient settings
    bf16=True,
    tf32=True,
    max_grad_norm=0.3, 
    warmup_ratio=0.05,
    # Hub and reporting
    push_to_hub=False,
    report_to="wandb" if args.wandb else 'none',
    # Gradient checkpointing settings
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # Dataset configuration
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    remove_unused_columns=False,
    dataloader_num_workers=16,
    dataloader_persistent_workers=True,
    # Seed
    seed=args.seed,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    data_collator=train_collator,
    peft_config=peft_config if args.lora else None,
)

trainer.train()

trainer.save_model(training_args.output_dir)
