"""This script creates self edits.

Recall from the paper that the general definition of a self-edit is arbitrary
code that transforms an old system into a new system. However, for practical 
reasons, SEAL (and our study) restrict self-edits to adapt the model via a fixed 
update operator (self-supervised next token prediction training using LoRA). Our
study gives the model freedom to choose its training data D, and the values of 
selected hyperparameters H. Thus the model is searching over the space of (D,H) 
pairs.

For our methods, this script performs the "CreateSelfEditTemplates" and 
"CompleteSelfEditTemplates" steps in the paper. We
1. load SQuAD articles (either the 50-passage training subset of 200-passage
   validation one from SEAL),
2. generate self-edit templates (data creation instructions + hyperparameters H),
3. for each template and article, complete the template to fill in the training
   data D.

For the SEAL baseline, this script performs only "CompleteSelfEditTemplates" as
the data creation instruction is hardcoded & comes from SEAL. We
1. load SQuAD articles,
2. for each article, complete the hardcoded data creation instruction to get D.
3. use fixed hyperparameters H from SEAL.

"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import random
import re
import time
from pathlib import Path
from string import Template
from typing import Any, Dict, List

import requests
from pydantic import BaseModel

class Hyperparameters(BaseModel):
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    num_epochs: int
    gradient_accumulation_steps: int

class SelfEditTemplate(BaseModel):
    data_creation_instruction: str
    hyperparameters: Hyperparameters

class TrainingSequences(BaseModel):
    training_sequences: list[str]


BASELINE_IMPLICATIONS_PROMPT = (
    "Let's read the following passage and produce a list of implications "
    "derived directly or indirectly from the text.\n\n"  
    "Passage:\n{title}\n{passage}\n\n"
    "Implications:\n"
)

BASELINE_REWRITES_PROMPT = (
    "Let's read the following passage and rewrite it in a few different ways, "
    "each one separated by a newline.\n\n"
    "Passage:\n{title}\n{passage}\n\n"
    "Rewritten passages:\n"
)

BASELINE_HYPERPARAMETERS = {
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0,
    "learning_rate": 0.001,
    "num_epochs": 10,
    "gradient_accumulation_steps": 1
}


def strip_thinking_content(text: str) -> str:
    """Remove content within <think>...</think> tags."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def parse_self_edit_template(response: str) -> tuple[bool, Dict[str, Any] | None, str]:
    """Parse a template response as JSON and validate the expected schema.

    Args:
        response: Raw model response string.

    Returns:
        A tuple of (is_valid, parsed_json, error_message).
    """
    # Strip thinking content if present
    response = strip_thinking_content(response)
    response = response.strip()

    # Try to extract JSON from response (in case there's extra text)
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if not json_match:
        return False, None, "No JSON object found in response"
    
    json_str = json_match.group(0)
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return False, None, f"JSON parsing error: {e}"
    
    # Validate structure
    required_keys = {"data_creation_instruction", "hyperparameters"}
    if not required_keys.issubset(data.keys()):
        return False, data, f"Missing required keys. Expected {required_keys}, got {data.keys()}"
    
    hyperparams = data.get("hyperparameters", {})
    required_hyperparam_keys = {
        "lora_rank", "lora_alpha", "lora_dropout", 
        "learning_rate", "num_epochs", 
        "gradient_accumulation_steps"
    }
    
    if not required_hyperparam_keys.issubset(hyperparams.keys()):
        return False, data, f"Missing hyperparam keys. Expected {required_hyperparam_keys}, got {hyperparams.keys()}"
    
    # Validate types
    try:
        assert isinstance(data["data_creation_instruction"], str)
        assert isinstance(hyperparams["lora_rank"], int)
        assert isinstance(hyperparams["lora_alpha"], int)
        assert isinstance(hyperparams["lora_dropout"], (float, int))
        assert isinstance(hyperparams["learning_rate"], (float, int))
        assert isinstance(hyperparams["num_epochs"], int)
        assert isinstance(hyperparams["gradient_accumulation_steps"], int)
    except AssertionError:
        return False, data, "Type validation failed"
    
    return True, data, "Valid"

def generate_self_edit_templates(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Generate self-edit templates using the meta-prompt and optional archive.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A list of dictionaries. Each dictionary is one self-edit template, which
        contains a data creation instruction, hyperparameters, and other metadata.
    """
    if args.num_self_edit_templates <= 0:
        raise ValueError("num_self_edit_templates must be > 0")

    if args.baseline:
        assert args.num_self_edit_templates == 1, "Baseline run can only have one template"
        prompt = BASELINE_IMPLICATIONS_PROMPT if args.baseline == "implications" else BASELINE_REWRITES_PROMPT
        return [{
            "data_creation_instruction": prompt,
            "hyperparameters": BASELINE_HYPERPARAMETERS,
            "full_message": None
        }]

    # Everything from here on out is non-baseline.
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

    messages = [{"role": "user", "content": meta_prompt}]

    if args.thinking_mode:
        # Exploit (conservative) decoding parameters
        params_exploit = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": -1,
            "min_p": 0.05,
            "presence_penalty": 0.0
        }

        # Explore (creative) decoding parameters
        params_explore = {
            "temperature": 1.3,
            "top_p": 0.95,
            "top_k": -1,
            "min_p": 0.1,
            "presence_penalty": 0.0
        }
    else:
        # Exploit (conservative) decoding params when not in thinking mode
        params_exploit = {
            "temperature": 0.3,
            "top_p": 0.85,
            "top_k": 40,
            "min_p": 0.05,
            "presence_penalty": 0.5
        }

        # Explore (creative) decoding params when not in thinking mode
        params_explore = {
            "temperature": 1.3,
            "top_p": 0.95,
            "top_k": -1,
            "min_p": 0.05,
            "presence_penalty": 1.5
        }

    # Determine how many templates are exploit vs explore based on a ratio
    # Exploit fraction parameter is provided by the user (defaults to 0.6)
    num_exploit = int(round(args.num_self_edit_templates * args.exploit_fraction))
    # If the user explicitly set exploit_fraction to 0, allow num_exploit to be 0.
    # If exploit_fraction > 0 but rounding produced 0 (e.g., small fractions),
    # ensure at least one exploit template is generated to match original intent.
    if args.num_self_edit_templates > 0 and num_exploit == 0 and args.exploit_fraction > 0:
        num_exploit = 1
    if num_exploit > args.num_self_edit_templates:
        num_exploit = args.num_self_edit_templates
    num_explore = args.num_self_edit_templates - num_exploit
    payload_exploit = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_new_tokens,
        "temperature": params_exploit["temperature"],
        "top_p": params_exploit["top_p"],
        "top_k": params_exploit["top_k"],
        "min_p": params_exploit["min_p"],
        "presence_penalty": params_exploit["presence_penalty"],
        "n": num_exploit,
        "guided_json": SelfEditTemplate.model_json_schema(),
        "chat_template_kwargs": {"enable_thinking": True if args.thinking_mode else False},
    }
    payload_explore = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_new_tokens,
        "temperature": params_explore["temperature"],
        "top_p": params_explore["top_p"],
        "top_k": params_explore["top_k"],
        "min_p": params_explore["min_p"],
        "presence_penalty": params_explore["presence_penalty"],
        "n": num_explore,
        "guided_json": SelfEditTemplate.model_json_schema(),
        "chat_template_kwargs": {"enable_thinking": True if args.thinking_mode else False},
    }

    print(
        "Requesting "
        f"{args.num_self_edit_templates} self edit templates "
        f"(exploit={num_exploit}, explore={num_explore}) from "
        f"{args.vllm_api_url}/v1/chat/completions..."
    )
    t0 = time.time()

    exploit_choices = []
    explore_choices = []

    # Only send requests for non-zero counts to the API (avoid sending n=0)
    if num_exploit > 0:
        r = requests.post(f"{args.vllm_api_url}/v1/chat/completions", json=payload_exploit, timeout=3000)
        r.raise_for_status()
        exploit_choices = r.json()["choices"]

    if num_explore > 0:
        r = requests.post(f"{args.vllm_api_url}/v1/chat/completions", json=payload_explore, timeout=3000)
        r.raise_for_status()
        explore_choices = r.json()["choices"]

    # Build a unified list of tuples (content, full_message_dict)
    responses = []
    for choice in exploit_choices + explore_choices:
        full_message = choice.get("message", {})
        content = full_message.get("content", "").strip()
        responses.append((content, full_message))

    total_elapsed = time.time() - t0
    print(f"Received {len(responses)} self edit templates in {total_elapsed:.1f}s")

    self_edit_templates = []
    for idx, (response_content, full_message) in enumerate(responses):
        category = "exploit" if idx < len(exploit_choices) else "explore"
        is_valid, parsed_self_edit_template_dict, error_msg = parse_self_edit_template(response_content)
        if is_valid:
            print("Valid self edit template received.")
            print(f"Data Creation Instruction: {parsed_self_edit_template_dict['data_creation_instruction']}")
            print(f"Hyperparameters: {parsed_self_edit_template_dict['hyperparameters']}")
            # Store the full message from the model along with the parsed template
            parsed_self_edit_template_dict["full_message"] = full_message
            parsed_self_edit_template_dict["category"] = category
            self_edit_templates.append(parsed_self_edit_template_dict)
        else:
            print(f"Invalid self edit template response: {error_msg}")
            print(f"Response was: {response_content}\nFull message: {full_message}\n")
    
    return self_edit_templates

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

def parse_completion_baseline(response: str) -> Dict[str, List[str]]:
    """Parse a baseline response as a list of strings.

    Args:
        response: Raw model response string.

    Returns:
        A dict with a single key, "training_sequences".
    """
    response = strip_thinking_content(response)
    response = response.strip()
    arr = response.split("\n")
    # Remove leading lines when the model uses numbered list formatting.
    # This kind of text preprocessing mirrors what is seen in SEAL's codebase:
    # https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/general-knowledge/src/utils.py#L209
    if len(arr) > 1 and arr[1].startswith("1."):
        arr = arr[1:]
    elif len(arr) > 2 and arr[2].startswith("1."):
        arr = arr[2:]
    arr = [s.strip() for s in arr if s.strip()]
    return {"training_sequences": arr}

def parse_completion(response: str) -> tuple[bool, Dict[str, List[str]] | None, str]:
    """Parse a completion response as JSON.

    Args:
        response: Raw model response string.

    Returns:
        A tuple of (is_valid, parsed_json, error_message).
    """
    # Strip thinking content if present
    response = strip_thinking_content(response)
    response = response.strip()

    if not response.endswith('}'):
        response += '\n}'

    # Try to extract JSON from response (in case there's extra text)
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if not json_match:
        return False, None, "No JSON object found in response"
    
    json_str = json_match.group(0)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return False, None, f"JSON parsing error: {e}, the JSON string was: {json_str}"
    
    return True, data, "Valid"

def complete_self_edit_templates(
    args: argparse.Namespace,
    data_creation_instruction_dicts: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Generate completions for each data-creation instruction.

    Args:
        args: Parsed CLI arguments.
        data_creation_instruction_dicts: Per-article instructions and metadata.

    Returns:
        A list of completion dicts aligned with the input list order.
    """
    # Compute per-template counts for exploit vs explore completions
    num_exploit_per_template = int(round(args.num_completions_per_template * args.exploit_fraction))
    # If the user explicitly set exploit_fraction to 0, allow num_exploit_per_template to be 0.
    # If exploit_fraction > 0 but rounding produced 0 (e.g., small fractions),
    # ensure at least one exploit completion is generated to match original intent.
    if args.num_completions_per_template > 0 and num_exploit_per_template == 0 and args.exploit_fraction > 0:
        num_exploit_per_template = 1
    if num_exploit_per_template > args.num_completions_per_template:
        num_exploit_per_template = args.num_completions_per_template

    def process_item(d):
        if args.baseline:
            prompt = d["data_creation_instruction"].format(
                title=d["title"],
                passage=d["passage"],
            )
        else:
            prompt = COMPLETE_SELF_EDIT_TEMPLATE_PROMPT.format(
                title=d["title"],
                passage=d["passage"],
                data_creation_instruction=d["data_creation_instruction"]
            )
        messages = [{"role": "user", "content": prompt}]
        
        completion_idx = d.get("completion_idx_in_template", 0)
        is_exploit = completion_idx < num_exploit_per_template

        if args.thinking_mode:
            # Exploit (conservative) decoding parameters
            params_exploit = {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": -1,
                "min_p": 0.05,
                "presence_penalty": 0.0
            }

            # Explore (creative) decoding parameters
            params_explore = {
                "temperature": 1.3,
                "top_p": 0.95,
                "top_k": -1,
                "min_p": 0.1,
                "presence_penalty": 0.1
            }
        else:
            params_exploit = {
                "temperature": 0.3,
                "top_p": 0.85,
                "top_k": 40,
                "min_p": 0.05,
                "presence_penalty": 0.5
            }

            params_explore = {
                "temperature": 1.3,
                "top_p": 0.95,
                "top_k": -1,
                "min_p": 0.05,
                "presence_penalty": 1.5
            }

        # Choose parameters for this specific call.
        chosen_params = params_exploit if is_exploit else params_explore
        temperature = chosen_params["temperature"]
        top_p = chosen_params["top_p"]
        top_k = chosen_params["top_k"]
        min_p = chosen_params["min_p"]
        presence_penalty = chosen_params["presence_penalty"]

        if args.baseline:
            payload = {
                "model": args.model,
                "messages": messages,
                "max_tokens": args.max_new_tokens,
                "temperature": 1.0,
                "top_p": 0.95,
            }
        else:
            payload = {
                "model": args.model,
                "messages": messages,
                "max_tokens": args.max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "presence_penalty": presence_penalty,
                "chat_template_kwargs": {"enable_thinking": True if args.thinking_mode else False},
            }
            # We request one completion per call (these are parallelized).
            payload["n"] = 1
            payload["guided_json"] = TrainingSequences.model_json_schema()
        
        try:
            # Timeout is in seconds; 300s allows longer chat/streaming responses.
            r = requests.post(f"{args.vllm_api_url}/v1/chat/completions", json=payload, timeout=300)
            r.raise_for_status()
            message = r.json()["choices"][0]["message"]
            return {"content": message.get("content", "").strip(), "full_message": message}
        except Exception as e:
            print(f"Error processing item: {e}")
            return None

    print(f"Requesting {len(data_creation_instruction_dicts)} completions from {args.vllm_api_url}/v1/chat/completions (parallel)...")
    t0 = time.time()
    
    # Use ThreadPoolExecutor for parallel requests with a workload-aware limit.
    desired_workers = min(32, max(1, len(data_creation_instruction_dicts)))
    print(f"Using ThreadPoolExecutor with max_workers={desired_workers}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=desired_workers) as executor:
        responses = list(executor.map(process_item, data_creation_instruction_dicts))

    total_elapsed = time.time() - t0
    print(f"Received {len(responses)} completions in {total_elapsed:.1f}s")
    
    completions = []
    for response in responses:
        if response is None:
            completions.append({"training_sequences": [], "full_message": None})
            continue
        
        content = response["content"]
        full_message = response.get("full_message")
        if args.baseline:
            parsed = parse_completion_baseline(content)
            parsed["full_message"] = full_message
            completions.append(parsed)
        else:
            is_valid, parsed_training_sequences, error_msg = parse_completion(content)
            if is_valid:
                parsed_training_sequences["full_message"] = full_message
                completions.append(parsed_training_sequences)
            else:
                print(f"Invalid completion response: {error_msg}")
                print(f"Response was: {content}\nFull message: {full_message}\n")
                completions.append({"training_sequences": [], "full_message": full_message})

    return completions

def main():
    """CLI entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--vllm_api_url", required=True, help="e.g. http://localhost:8001")
    p.add_argument("--model", required=True, default="Qwen/Qwen3-8B", help="HF model name")
    p.add_argument("--instruct_model", action="store_true", help="Set this flag if you are using a Qwen instruct model")
    p.add_argument("--thinking_mode", action="store_true", help="Set this flag to enable thinking mode for Qwen3 models")
    p.add_argument("--dataset_in", required=True, help="Path to the input dataset")
    p.add_argument("--dataset_out", required=True, help="Path to the output dataset")
    p.add_argument("--archive_path", help="Path to the archive JSON for for archive evolution")
    p.add_argument("--meta_prompt_txt_path", help="Path to the meta-prompt text file used to generate self edit templates")
    p.add_argument("--num_articles", type=int, default=50, help="How many SQuAD articles to process")
    p.add_argument("--start_index", type=int, default=0, help="Start index for processing")
    p.add_argument("--num_self_edit_templates", type=int, default=5, help="Number of self edit templates to generate")
    p.add_argument("--num_completions_per_template", type=int, default=3, help="Number of completions per self edit template per article")
    p.add_argument("--max_new_tokens", type=int, default=8192, help="Max number of new tokens to generate after the prompt")
    p.add_argument("--exploit_fraction", type=float, default=0.6, help="Fraction (0..1) of self edit templates that are exploit (conservative) vs explore (creative)")
    p.add_argument("--baseline", choices=['implications', 'rewrite'], help="Set this flag to run baselines")
    args = p.parse_args()

    # Load SQuAD data and take subset
    with open(args.dataset_in, encoding="utf-8") as fh:
        raw: List[Dict[str, Any]] = json.load(fh)
    random.seed(42)  # Fixed seed for reproducibility. To sample a different subset, change args.start
    random.shuffle(raw)
    if args.num_articles > 0:
        subset = raw[args.start_index : args.start_index + args.num_articles]
    else:
        subset = raw[args.start_index:]

    # Generate self edit templates
    self_edit_templates = generate_self_edit_templates(args)

    # For each template, generate `num_completions_per_template` completions per article.
    # Each completion is an instantiation of the template.
    data_creation_instruction_dicts = []
    for t_idx, self_edit_template in enumerate(self_edit_templates):
        data_creation_instruction = self_edit_template["data_creation_instruction"]
        for squad_article in subset:
            for completion_idx in range(args.num_completions_per_template):
                data_creation_instruction_dicts.append({
                    "data_creation_instruction": data_creation_instruction,
                    "title": squad_article["title"],
                    "passage": squad_article["context"],
                    "template_idx": t_idx,
                    "completion_idx_in_template": completion_idx,
                })
    completions = complete_self_edit_templates(args, data_creation_instruction_dicts)

    assert len(completions) == len(data_creation_instruction_dicts), "Number of completions does not match number of data creation instructions requested"

    # Organize output data.
    """
    Output data structure:
    {
        "metadata": {
            "args": { All command line args used to generate this dataset },
            "timestamp": ISO 8601 timestamp string
        },
        "articles": [ Each element in this array corresponds to 1 article. There are a total of `num_articles` articles.
            {
                "title": "Article 1 Title",
                "context": "Full text of article 1...",
                "questions": [
                    {
                            "question": "What is the main topic of article 1?",
                            "answer": "answer text..."
                    },
                    ...
                ]
            },
            ...        
        ],
        "self_edit_templates": [ Each element in this array corresponds to 1 self edit template. There are a total of `num_self_edit_templates` templates.
            {
                "data_creation_instruction": string,
                "hyperparameters": {
                    "lora_rank": int,
                    "lora_alpha": int,
                    "lora_dropout": float,
                    "learning_rate": float,
                    "num_epochs": int,
                    "gradient_accumulation_steps": int
                },
                "full_message": object (optional, the raw model message dict returned when generating this template, only present when thinking_mode is true),
                "category": string in {"exploit","explore"} (optional, indicates whether this template was generated under conservative or creative sampling parameters),
                "completions": {
                    # Each key is the index of the article in the "articles" array above.
                    # Each value is an array of `num_completions_per_template` completions for that article, that follow the current self edit template.
                    # Each completion is itself an array of training sequences (strings).
                    "article_index (int)": [
                            {"training_sequences": [ "training sequence 1", "training sequence 2", ... ], "full_message": { ... } },  # Completion 1 for this article
                            {"training_sequences": [ "training sequence 1", "training sequence 2", ... ], "full_message": { ... } },  # Completion 2 for this article
                            ...
                    ],
                    ...
                }
            },
            # FYI: The number of self edits is num_self_edit_templates * num_completions_per_template. Each completion is like an instantiation of the self edit template.
            ...
         ]
    }
    """
    articles_data = []
    for squad_article in subset:
        articles_data.append({
            "title": squad_article["title"],
            "context": squad_article["context"],
            "questions": squad_article["questions"]
        })

    self_edits_data = []
    for self_edit_template in self_edit_templates:
        template_completions = {}
        for article_idx in range(len(subset)):
            article_completions = []
            for _ in range(args.num_completions_per_template):
                article_completion = completions.pop(0)
                training_sequences = article_completion["training_sequences"]
                # Keep the full output message (reasoning trace) with each completion
                article_completions.append({
                    "training_sequences": training_sequences,
                    "full_message": article_completion.get("full_message")
                })
            template_completions[article_idx] = article_completions
        
        self_edits_data.append({
            "data_creation_instruction": self_edit_template["data_creation_instruction"],
            "hyperparameters": self_edit_template["hyperparameters"],
            "full_message": self_edit_template.get("full_message"),
            "category": self_edit_template.get("category"),
            "completions": template_completions
        })

    # Create metadata
    meta = {
        "args": vars(args),
        # Use timezone-aware UTC timestamp
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
    }

    output_data = {
        "metadata": meta,
        "articles": articles_data,
        "self_edit_templates": self_edits_data
    }

    output_path = Path(args.dataset_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write the output file safely with explicit close
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output_data, fh, ensure_ascii=False, indent=2)
    print(f"Output data saved to {output_path}")

if __name__ == "__main__":
    main()