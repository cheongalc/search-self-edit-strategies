import argparse
import json
import os
import re
import requests
import concurrent.futures
from datetime import datetime
from pathlib import Path
from string import Template
from typing import List, Dict, Any

COMPLETE_SELF_EDIT_TEMPLATE_PROMPT = (
    "Consider the following passage:\n\n"
    "[BEGINNING OF PASSAGE]\n"
    "{title}\n{passage}\n\n"
    "[END OF PASSAGE]\n\n"
    "Now, refer to the following data creation instruction, which defines how the above passage shall be transformed into a list of one or more training sequences:\n"
    "[BEGINNING OF INSTRUCTION]\n"
    "{data_creation_instruction}\n"
    "[END OF INSTRUCTION]\n\n"
    "You must follow the data creation instruction carefully to generate the training sequences. Your training sequences must NEVER be empty.\n"
    "You must output a single JSON object. Do not output anything outside the JSON. The JSON must follow this structure: {{\"training_sequences\": list of strings}}."
)

def extract_cot(full_message: Dict[str, Any]) -> str:
    """Extracts CoT from the full message. Checks 'reasoning_content' first, then <think> tags in content."""
    if "reasoning_content" in full_message and full_message["reasoning_content"]:
        return full_message["reasoning_content"]
    
    content = full_message.get("content", "")
    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return ""

def call_vllm_batch(prompts: List[str], vllm_url: str, model: str, max_tokens: int = 1024) -> List[str]:
    """Sends a batch of prompts to VLLM using threading."""
    
    def _call_one(prompt):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 1.3,
            "top_p": 0.95,
            "top_k": -1,
            "min_p": 0.1,
            "presence_penalty": 0.0
        }
        try:
            r = requests.post(f"{vllm_url}/v1/chat/completions", json=payload, timeout=300)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error calling VLLM: {e}")
            return ""

    # Use ThreadPoolExecutor
    workers = min(32, len(prompts))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_call_one, prompts))
    
    return results

def build_dataset_ours(args: argparse.Namespace) -> str:
    self_edit_results_json = json.load(open(args.self_edit_results_json_path, encoding="utf-8"))

    timestamp = self_edit_results_json.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_output_dir = Path(args.dataset_output_dir)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    dataset_output_path = os.path.join(dataset_output_dir, f"sft_dataset_{timestamp}_top{args.k_best}.jsonl")

    assert args.meta_prompt_txt_path is not None, "Cannot have empty meta prompt path with custom self edits"
    meta_prompt_path = Path(args.meta_prompt_txt_path)
    meta_prompt = meta_prompt_path.read_text()

    if args.archive_path:
        archive_path = Path(args.archive_path)
        with open(archive_path, "r") as f:
            archive = json.load(f)
            assert len(archive) >= 4, "Archive has incorrect length"
    
            template_contents = [{
                "data_creation_instruction": t["data_creation_instruction"],
                "hyperparameters": t["hyperparameters"]
            } for t in archive]

            meta_prompt_template = Template(meta_prompt)
            meta_prompt = meta_prompt_template.substitute(
                highest_acc_template=json.dumps(template_contents[0]),
                acc1 = archive[0]["accuracy"],
                ngain1 = archive[0]["normalized_gain"],
                second_highest_acc_template=json.dumps(template_contents[1]),
                acc2 = archive[1]["accuracy"],
                ngain2 = archive[1]["normalized_gain"],
                second_lowest_acc_template=json.dumps(template_contents[-2]),
                acc3 = archive[-2]["accuracy"],
                ngain3 = archive[-2]["normalized_gain"],
                lowest_acc_template=json.dumps(template_contents[-1]),
                acc4 = archive[-1]["accuracy"],
                ngain4 = archive[-1]["normalized_gain"],
            )

    with open(dataset_output_path, "w") as out:
        for article_index, article_results in self_edit_results_json["article_statistics"].items():
            template_data = [
                (template_index, template_results["self_edit_mean_acc"])
                for template_index, template_results in article_results["self_edit_template_statistics"].items()
            ]
            # Sort the self edit templates according to what their best mean accuracy is, and select the top k_best templates.
            # Note: In our method, we have multiple self edit templates per article, and each template has multiple completions.
            # For each article, it is naive to just pick the top scoring completion overall. Because:
            # 1. A completion could have turned out well because of a very strong template, which boosted the performance even if the completion was mediocre.
            # 2. The template could have been mediocre but the completion was outstanding.
            # So, we first pick the top k_best templates for each article, and then from each template we pick the best completion.
            # We pick the top k_best templates by first computing the usefulness of each template. This is the mean gain in accuracy across all completions for that template.
            # But note that computing the mean gain in accuracy and sorting is the same as computing the main accuracy and sorting, because the gain is just the accuracy minus a constant (from the baseline).
            # Thus it is permissible to just sort by the mean accuracy directly.
            best_templates_data = sorted(template_data, key=lambda x: x[1], reverse=True)[:args.k_best]
            
            # If archive exists, filter out templates that did not yield a positive gain
            if args.archive_path:
                baseline_acc = article_results.get("baseline_mean_acc", 0)
                best_templates_data = [(t_idx, t_acc) for t_idx, t_acc in best_templates_data if t_acc > baseline_acc]
            
            # Now that we've picked the best templates, we need to pick the best completion from each template.
            best_completion_indices = []
            for template_index, _ in best_templates_data:
                completions_data = [
                    (completion_index, completion_results["self_edit_mean_acc"])
                    for completion_index, completion_results in article_results['self_edit_template_statistics'][template_index]["completions"].items()
                ]
                # If archive exists, filter out completions that did not yield a positive gain
                if args.archive_path:
                    baseline_acc = article_results.get("baseline_mean_acc", 0)
                    completions_data = [(c_idx, c_acc) for c_idx, c_acc in completions_data if c_acc > baseline_acc]
                
                if completions_data:
                    best_completion_idx, _ = max(completions_data, key=lambda x: x[1])
                    best_completion_indices.append(best_completion_idx)
                else:
                    best_completion_indices.append(None)
        
            for (best_template_index, _), best_completion_index in zip(best_templates_data, best_completion_indices):
                # Skip if no valid completion was found for this template
                if best_completion_index is None:
                    continue
                raw_best_template = self_edit_results_json["dataset"]["self_edit_templates"][int(best_template_index)]
                training_seq_to_teach_how_to_write_templates = {
                    "prompt": meta_prompt,
                    "completion": json.dumps({"data_creation_instruction": raw_best_template["data_creation_instruction"], "hyperparameters": raw_best_template["hyperparameters"]}, ensure_ascii=False)
                }
                out.write(json.dumps(training_seq_to_teach_how_to_write_templates, ensure_ascii=False) + "\n")
                
                raw_best_completion = raw_best_template["completions"][article_index][int(best_completion_index)]
                training_seq_to_teach_how_to_fill_templates = {
                    "prompt": COMPLETE_SELF_EDIT_TEMPLATE_PROMPT.format(
                        title=self_edit_results_json["dataset"]["articles"][int(article_index)]["title"],
                        passage=self_edit_results_json["dataset"]["articles"][int(article_index)]["context"],
                        data_creation_instruction=raw_best_template["data_creation_instruction"]
                    ),
                    "completion": json.dumps({"training_sequences": raw_best_completion["training_sequences"]}, ensure_ascii=False)
                }
                out.write(json.dumps(training_seq_to_teach_how_to_fill_templates, ensure_ascii=False) + "\n")
        
        print(f"Finished writing SFT dataset to {dataset_output_path}.")
        return dataset_output_path

def build_dataset_baseline(args: argparse.Namespace):
    self_edit_results_json = json.load(open(args.self_edit_results_json_path, encoding="utf-8"))

    timestamp = self_edit_results_json.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_output_dir = Path(args.dataset_output_dir)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    dataset_output_path = os.path.join(dataset_output_dir, f"sft_dataset_{timestamp}_top{args.k_best}.jsonl")

    with open(dataset_output_path, "w") as out:
        for article_index, article_results in self_edit_results_json["article_statistics"].items():
            assert "self_edit_template_statistics" in article_results, f"Results for article {article_index} missing self edit template statistics!"
            # Check that the number of self edit templates should be exactly 1, because in the baseline we only generate 1 template per article
            # Note that article_results["self_edit_template_statistics"] is a dict where keys are template indices (as strings) and values are the statistics for that template
            assert len(article_results["self_edit_template_statistics"]) == 1, f"Expected exactly 1 self edit template for article {article_index}, but found {len(article_results['self_edit_template_statistics'])}!"
            
            # Grab the top k_best completions for just that one template.
            completions = [
                (c_idx, c_data["self_edit_mean_acc"])
                for c_idx, c_data in article_results['self_edit_template_statistics']["0"]["completions"].items()
            ]
            # Sort according to the mean accuracy of that completion.
            best_completions = sorted(completions, key=lambda x: x[1], reverse=True)[:args.k_best]

            # Dump the best completions to the output file.
            for c_idx, _ in best_completions:
                training_seq = {
                    # Recall that for the baseline, the data creation instruction looks something like: "Let's read the following passage and produce a list of implications derived directly or indirectly from the text.\n\nPassage:\n{title}\n{passage}\n\nImplications:\n"
                    # The {title} and {passage} placeholders have NOT been filled in yet!
                    # So that's why we fill it in here.
                    "prompt": self_edit_results_json["dataset"]["self_edit_templates"][0]["data_creation_instruction"].format(
                        title=self_edit_results_json["dataset"]["articles"][int(article_index)]["title"],
                        passage=self_edit_results_json["dataset"]["articles"][int(article_index)]["context"]
                    ),
                    "completion": self_edit_results_json["dataset"]["self_edit_templates"][0]["completions"][article_index][int(c_idx)]["full_message"]["content"]
                }
                out.write(json.dumps(training_seq, ensure_ascii=False) + "\n")

    print(f"Finished writing SFT dataset to {dataset_output_path}.")
    return dataset_output_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self_edit_results_json_path", required=True, help="Path to the result JSON from the apply self edits stage")
    p.add_argument("--dataset_output_dir", required=True, help="Destination folder to put the SFT dataset")
    p.add_argument("--meta_prompt_txt_path", help="Path to the text file containing the meta prompt that is used to make the model generate Learning Plan Templates")
    p.add_argument("--k_best", type=int, default=2, help="Baseline: Select top-k completions per article. Our method: Select top-k templates per article and 1 completion from each template")
    p.add_argument("--baseline", action="store_true", help="Whether to use baseline")
    p.add_argument("--vllm_api_url", help="VLLM API URL for distillation")
    p.add_argument("--model", help="Model name for distillation")
    p.add_argument("--archive_path", help="Path to archive JSON file (if needed)")
    args = p.parse_args()

    if args.baseline:
        build_dataset_baseline(args)
    else:
        build_dataset_ours(args)

if __name__ == "__main__":
    main()