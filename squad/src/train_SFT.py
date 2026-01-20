"""
SFT Trainer.

Dataset format expected:
{"prompt": "...", "completion": "..."}
"""
import os
import argparse
from datasets import load_dataset
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_dataset_path", required=True, help="Path to the SFT dataset in JSONL format")
    p.add_argument("--model_name", required=True, help="Name or path of the pre-trained model to fine-tune")
    p.add_argument("--checkpoint_output_dir", required=True, help="Directory to save the fine-tuned model checkpoint")
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=5)
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--training_max_seq_length", type=int, default=0, help="Max token length to use for training; 0 means compute from dataset")
    return p.parse_args()

def longest_seq_len(dataset, tok):
    return max(
        len(tok(example["prompt"] + example["completion"]).input_ids)
        for example in dataset
    )

def main() -> None:
    args = parse_args()

    dataset = load_dataset("json", data_files=args.sft_dataset_path, split="train")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Compute a robust max_length for SFT training; avoid using single extreme outlier
    max_len_from_data = longest_seq_len(dataset, tokenizer)
    if args.training_max_seq_length and args.training_max_seq_length > 0:
        chosen_max_length = min(max_len_from_data, args.training_max_seq_length)
    else:
        chosen_max_length = max_len_from_data

    print(f"Dataset max token length: {max_len_from_data}; Training max seq length chosen: {chosen_max_length}")

    # Truncate long examples in dataset if we've chosen a cap
    if chosen_max_length and chosen_max_length > 0:
        print("Truncating dataset examples to chosen max seq length...")
        def _truncate_example(example):
            prompt = example.get("prompt", "") or ""
            completion = example.get("completion", "") or ""
            # tokenization without special tokens
            p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            c_ids = tokenizer(completion, add_special_tokens=False).input_ids
            total = len(p_ids) + len(c_ids)
            if total <= chosen_max_length:
                return example
            allowed_comp = chosen_max_length - len(p_ids)
            # ensure at least 1 token for completion; if not, shorten prompt instead
            if allowed_comp < 1:
                # allocate half-half if prompt alone exceeds budget
                allowed_prompt = max(1, chosen_max_length // 2)
                allowed_comp = chosen_max_length - allowed_prompt
                p_ids = p_ids[:allowed_prompt]
                prompt = tokenizer.decode(p_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            c_ids = c_ids[:max(allowed_comp, 0)]
            new_comp = tokenizer.decode(c_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            example["completion"] = new_comp
            return example

        try:
            dataset = dataset.map(_truncate_example)
        except Exception as e:
            print("Warning: dataset truncation failed with error:", str(e))
    
    sft_args = SFTConfig(
        output_dir=args.checkpoint_output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        max_length=chosen_max_length,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules.split(","),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_cfg,
    )

    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    trainer.train()
    peft_model = trainer.model
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(args.checkpoint_output_dir)
    tokenizer.save_pretrained(args.checkpoint_output_dir)
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
