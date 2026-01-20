"""Apply self edits end-to-end on a single node.

This implementation mirrors the orchestration logic shown in
``general-knowledge/src/query/query_server.py`` and
``general-knowledge/src/inner/TTT_server.py`` of the original SEAL repo, but
stays self-contained inside 1 file.

Recall from the paper that the process of applying self edits looks like:
For each self edit:
1. Load the self-edit (which includes the training data and hyperparameters we
   will use to adapt the model).
2. Clone the base model and train this copy using a LoRA adapter on the self-edit's
   training data and hyperparameters.
3. Evaluate the baseline model and the adapted model on the SQuAD questions.
4. Calculate the difference in QA accuracy.

This file does those 4 conceptual steps, but parallelizes across multiple GPUs.

It does the following: (you can see the overall pipeline in the run() method of 
the Orchestrator class):
1. Load the self-edit dataset. We call it a self edit dataset because it is a 
   file that contains all the self edits that are to be applied. It also
   contains the SQuAD articles that were used to generate the self edits, as
   well as the SQuAD questions that will be used to evaluate the self edits.
2. Train all the "adapted models" first. This is done by putting "executor"
   workers on each GPU, and having them pull LoRA training jobs from a queue.
3. Once all the adapted models are trained, fire up VLLM servers on each GPU.
4. Now do all the baseline evaluations. This calculates the QA accuracy of the 
   base model BEFORE the application of any self edits!
5. Finally evaluate all the "adapted models".
6. Collect all results and write to an output file. Update the archive if needed.

While steps 1-6 are happening, the script also launches a simple HTTP server to
show progress. There are also convenience options to persist the intermediate
states of the script so that you can pick up from where you left off if the 
script dies halfway.
)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import functools
import gc
import hashlib
import http.server
import json
import logging
import multiprocessing as mp
import os
import queue
import random
import re
import requests
import shutil
import socketserver
import statistics as _stats
import subprocess
import sys
import threading
import time
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Literal, Union, Tuple, Sequence

import requests
import torch
from dotenv import load_dotenv
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from anthropic import Anthropic

DEFAULT_NUM_GPUS = 8
DEFAULT_MAX_SEQ_LEN_FOR_TRAINING_ADAPTERS = 2048
PROGRESS_REFRESH_SECS = 2.0
TRAINER_SEED_BASE = 1337

SQUAD_ANSWER_TEMPLATE_BASE = (
    "Let's answer a question directly and concisely.\n"
    "Question: {question}\n"
    "Answer:\n"
)
SQUAD_ANSWER_TEMPLATE_BASE_COT = (
    "Let's think step by step and then answer directly. Provide reasoning under 'Reasoning:' "
    "and the final answer under 'Final answer:'.\n"
    "Question: {question}\n"
    "Reasoning:"
)
SQUAD_ANSWER_TEMPLATE_QWEN_INSTRUCT = (
    "<|im_start|>system\nYou are an assistant to answer a question directly and concisely.<|im_end|>\n"
    "<|im_start|>user\n{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
SQUAD_GRADE_TEMPLATE = (
    "You are a grading assistant. Determine whether the student's answer is correct based solely on the gold answer. "
    "Respond only with yes or no.\n\nQuestion: {question}\nGold answer: {gold}\nStudent answer: {pred}\n"
    "Is the student answer correct?"
)
_FINAL_ANSWER_RE = re.compile(r"(?:^|\n)\s*final\s*answer\s*[:\-]\s*(.*)", re.I)
_YES_RE = re.compile(r"\b(yes)\b", re.I)
_NO_RE = re.compile(r"\b(no)\b", re.I)


class Question(BaseModel):
    question: str
    answer: str


class Article(BaseModel):
    title: str
    context: str
    questions: List[Question]

class Hyperparameters(BaseModel):
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    num_epochs: int
    gradient_accumulation_steps: int
    batch_size: int = 1

class CompletionInstance(BaseModel):
    training_sequences: List[str] = []
    full_message: Optional[Dict[str, Any]] = None


def _normalize_self_edit_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy and new-format self edit payloads to a single schema."""

    templates = payload.get("self_edit_templates", []) or []
    for template in templates:
        completions = template.get("completions", {})
        if not isinstance(completions, dict):
            template["completions"] = {}
            continue

        normalized: Dict[str, List[Dict[str, Any]]] = {}
        for article_key, completion_list in completions.items():
            normalized_list: List[Dict[str, Any]] = []
            if isinstance(completion_list, list):
                for entry in completion_list:
                    if isinstance(entry, dict):
                        raw_sequences = entry.get("training_sequences")
                        if raw_sequences is None and "training_sequences" not in entry:
                            raw_sequences = entry.get("sequences")
                        if isinstance(raw_sequences, list):
                            sequences = list(raw_sequences)
                        elif raw_sequences is None:
                            sequences = []
                        else:
                            sequences = [str(raw_sequences)]
                        normalized_list.append(
                            {
                                "training_sequences": sequences,
                                "full_message": entry.get("full_message"),
                            }
                        )
                    elif isinstance(entry, list):
                        normalized_list.append({"training_sequences": list(entry)})
                    else:
                        normalized_list.append({"training_sequences": []})
            normalized[str(article_key)] = normalized_list
        template["completions"] = normalized

    return payload


class SelfEditTemplate(BaseModel):
    data_creation_instruction: str
    hyperparameters: Hyperparameters
    completions: Dict[str, List[CompletionInstance]]
    full_message: Optional[Dict[str, Any]] = None
    category: Optional[str] = None


class SelfEditDataset(BaseModel):
    metadata: Dict[str, Any]
    articles: List[Article]
    self_edit_templates: List[SelfEditTemplate]

    @classmethod
    def load(cls, path: Path) -> "SelfEditDataset":
        """Load a self edit dataset from a JSON file.

        Args:
            path: Path to the JSON file containing the dataset.

        Returns:
            An instance of SelfEditDataset loaded from the file.
        """
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        payload = _normalize_self_edit_payload(payload)
        return cls(**payload)

    @property
    def num_articles(self) -> int:
        return len(self.articles)

    @property
    def num_self_edit_templates(self) -> int:
        return len(self.self_edit_templates)

    @property
    def num_completions_per_template(self) -> int:
        for template in self.self_edit_templates:
            for _, completions in template.completions.items():
                return len(completions)
        return 0


def ensure_dir(path: Path) -> Path:
    """Ensure that a directory exists, creating it if necessary.

    Args:
        path: The directory path to ensure.

    Returns:
        The path to the directory.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def truncate(text: str, limit: int = 120) -> str:
    """Truncate text to a specified limit, appending ellipsis if truncated.

    Args:
        text: The text to truncate.
        limit: The maximum length of the text.

    Returns:
        The truncated text.
    """
    return text if len(text) <= limit else text[: limit - 3] + "..."


def deterministic_seed(key: str) -> int:
    """Generate a deterministic integer seed from a string key.

    Args:
        key: The string key to hash.

    Returns:
        An integer seed derived from the key.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def extract_final_answer(text: str) -> str:
    """Extract the final answer from a chain-of-thought response.

    Looks for patterns like "Final Answer:" or similar.

    Args:
        text: The model's output text.

    Returns:
        The extracted final answer, or "idk" if not found.
    """
    if not text:
        return "idk"
    match = _FINAL_ANSWER_RE.search(text)
    return (match.group(1).strip() if match else text.strip()) or "idk"


def format_answer_prompts(questions: List[Dict[str, str]], instruct: bool, chain_of_thought: bool) -> List[Union[str, List[Dict[str, str]]]]:
    """Format prompts for answering questions.

    Args:
        questions: List of dictionaries containing question data.
        instruct: Whether to use instruction tuning format.
        chain_of_thought: Whether to use chain-of-thought prompting.

    Returns:
        A list of formatted prompt strings or message lists.
    """
    if instruct:
        # Return message lists for the chat API.
        prompts = []
        for q in questions:
            msgs = [
                {"role": "system", "content": "You are an assistant to answer a question directly and concisely."},
                {"role": "user", "content": q["question"]}
            ]
            prompts.append(msgs)
        return prompts
    elif chain_of_thought:
        template = SQUAD_ANSWER_TEMPLATE_BASE_COT
    else:
        template = SQUAD_ANSWER_TEMPLATE_BASE
    return [template.format(question=q["question"]) for q in questions]


def format_grade_prompts(questions: List[Dict[str, str]], predictions: List[str]) -> List[str]:
    """Format prompts for grading answers.

    Args:
        questions: List of dictionaries containing question and gold answer.
        predictions: List of predicted answers.

    Returns:
        A list of formatted grading prompts.
    """
    return [
        SQUAD_GRADE_TEMPLATE.format(question=q["question"], gold=q["answer"], pred=p.strip())
        for q, p in zip(questions, predictions)
    ]


class AnswerGrader:
    """Grades answers against gold answers using heuristics or an LLM."""
    def __init__(self, grader_type: str = "heuristic", anthropic_model: str = "claude-haiku-4-5-20251001", env_file: Optional[Path] = None) -> None:
        """Initialize the AnswerGrader.

        Args:
            grader_type: The type of grader to use ("heuristic" or "anthropic").
            anthropic_model: The Anthropic model to use if grader_type is "anthropic".
            env_file: Path to an environment file containing API keys.
        """
        if env_file:
            load_dotenv(env_file)

        self.use_llm = (grader_type == "anthropic")
        if self.use_llm and Anthropic is None:
            logging.warning("Anthropic grader requested but 'anthropic' package not installed; falling back to heuristics")
            self.use_llm = False

        key = os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=key) if self.use_llm and key else None
        self.model = anthropic_model
        if self.use_llm and not self.client:
            logging.warning("LLM grading requested but ANTHROPIC_API_KEY missing; falling back to heuristics")
            self.use_llm = False

    def grade(self, questions: List[Dict[str, str]], predictions: List[str]) -> List[bool]:
        """Grade a list of predictions against the corresponding questions.

        Args:
            questions: List of question dictionaries containing gold answers.
            predictions: List of predicted answers.

        Returns:
            A list of boolean verdicts (True for correct, False for incorrect).
        """
        if not questions:
            return []
        if self.use_llm and self.client:
            prompts = format_grade_prompts(questions, predictions)
            verdicts: List[bool] = []
            for prompt in prompts:
                logging.debug("Grader Prompt: %s", prompt)
                verdict = self._query_llm(prompt)
                logging.debug("Grader Verdict: %s", verdict)
                verdicts.append(verdict)
            return verdicts
        return [self._heuristic(q["answer"], pred) for q, pred in zip(questions, predictions)]

    def _heuristic(self, gold: str, pred: str) -> bool:
        """Heuristic grading: exact match or substring match (case-insensitive).

        Args:
            gold: The gold answer.
            pred: The predicted answer.

        Returns:
            True if the prediction matches the gold answer, False otherwise.
        """
        gold_norm = (gold or "").strip().lower()
        pred_norm = (pred or "").strip().lower()
        if not gold_norm or not pred_norm:
            return False
        if gold_norm in pred_norm:
            return True
        return gold_norm == pred_norm

    def _query_llm(self, prompt: str) -> bool:
        """Query the LLM to grade an answer.

        Args:
            prompt: The grading prompt.

        Returns:
            True if the LLM judges the answer as correct, False otherwise.
        """
        assert self.client is not None
        max_retries = 5
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text if resp.content else ""
                logging.debug("Grader LLM Response: %s", text)
                if _YES_RE.search(text) and not _NO_RE.search(text):
                    return True
                if _NO_RE.search(text):
                    return False
            except Exception as exc:
                # Handle rate limits (HTTP 429).
                is_rate_limit = "429" in str(exc) or (hasattr(exc, "status_code") and exc.status_code == 429)
                
                if is_rate_limit:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.warning("LLM grading rate limit (429) on attempt %d/%d. Sleeping %.2fs...", attempt + 1, max_retries, delay)
                    time.sleep(delay)
                else:
                    logging.warning("LLM grading attempt %d/%d failed: %s", attempt + 1, max_retries, exc)
                    time.sleep(1.5 * (attempt + 1))
                    
        logging.error("LLM grading failed after %d attempts. Defaulting to False.", max_retries)
        return False


class AdapterTrainer:
    """Trains LoRA adapters for causal language models."""

    def __init__(self, model_name: str, instruct_model: bool, max_seq_length: int) -> None:
        """Initialize the AdapterTrainer.

        Args:
            model_name: Name or path of the base model.
            instruct_model: Whether the model is instruction-tuned.
            max_seq_length: Maximum sequence length for training.
        """
        # Disable HF tokenizer parallelism to avoid fork-related deadlocks in multiprocessing.
        # We already parallelize per GPU, so intra-process tokenizer threads are unnecessary.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.model_name = model_name
        self.instruct_model = instruct_model
        self.max_seq_length = max_seq_length
        logging.info("Loading base model %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use bfloat16 when supported; otherwise fall back to float16.
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        stop_token = "<|im_end|>" if instruct_model else (self.tokenizer.eos_token or "")
        self.stop_ids = self.tokenizer.encode(stop_token, add_special_tokens=False)
        self.collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)

    def train(
        self,
        self_edit_id: str,
        sequences: List[str],
        hyperparams: Dict[str, Any],
        scratch_dir: Path,
        gpu_id: int,
    ) -> Path:
        """Train a LoRA adapter on the provided sequences.

        Args:
            self_edit_id: Unique identifier for the self edit.
            sequences: List of training text sequences.
            hyperparams: Dictionary of hyperparameters.
            scratch_dir: Directory for temporary files.
            gpu_id: ID of the GPU to use.

        Returns:
            Path to the saved adapter.
        """
        rows = [self._encode_sequence(seq) for seq in sequences if seq.strip()]
        if not rows:
            raise RuntimeError("Training sequences are empty")
        dataset = HFDataset.from_list(rows)
        lora_cfg = LoraConfig(
            r=hyperparams["lora_rank"],
            lora_alpha=hyperparams["lora_alpha"],
            lora_dropout=hyperparams["lora_dropout"],
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        lora_model = get_peft_model(self.base_model, lora_cfg)
        self_edit_dir = ensure_dir(scratch_dir / f"gpu{gpu_id}" / self_edit_id)
        tmp_dir = ensure_dir(self_edit_dir / "trainer_state")
        training_args = TrainingArguments(
            output_dir=str(tmp_dir),
            per_device_train_batch_size=max(1, hyperparams.get("batch_size", 1)),
            gradient_accumulation_steps=max(1, hyperparams.get("gradient_accumulation_steps", 1)),
            num_train_epochs=max(1, hyperparams.get("num_epochs", 1)),
            learning_rate=float(hyperparams.get("learning_rate", 1e-3)),
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            fp16=False,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            seed=(TRAINER_SEED_BASE + deterministic_seed(self_edit_id)) & 0x7FFFFFFF,
            disable_tqdm=True,
        )
        trainer = Trainer(
            model=lora_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=self.collator,
        )
        trainer.train()
        adapter_path = ensure_dir(self_edit_dir / "final_adapter")
        lora_model.save_pretrained(str(adapter_path))
        del trainer, lora_model
        gc.collect()
        torch.cuda.empty_cache()
        return adapter_path

    def _encode_sequence(self, seq: str) -> Dict[str, List[int]]:
        """Tokenize and encode a sequence for training.

        Args:
            seq: The text sequence to encode.

        Returns:
            A dictionary containing input_ids, attention_mask, and labels.
        """
        # For instruct models, ensure the sequence ends with the stop token.
        if self.instruct_model and not seq.strip().endswith("<|im_end|>"):
            seq = seq + "<|im_end|>"

        tok = self.tokenizer(
            seq,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
        )
        # Standard CLM setup: labels mirror `input_ids` and are shifted internally.
        labels = list(tok["input_ids"])

        # Train on the full sequence; only pad tokens are masked (-100).
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels = [(-100 if t == pad_id else t) for t in labels]

        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "labels": labels,
        }


class VLLMClient:
    """Client for interacting with a vLLM API server."""

    def __init__(self, base_url: str, model_name: str, use_chat_api: bool = False, thinking_mode: bool = False) -> None:
        """Initialize the VLLMClient.

        Args:
            base_url: The base URL of the vLLM server.
            model_name: The name of the model served by vLLM.
            use_chat_api: Whether to use the chat completions API (required for instruct/reasoning models).
            thinking_mode: Whether to enable thinking mode (for Qwen 3).
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.use_chat_api = use_chat_api
        self.thinking_mode = thinking_mode
        self.session = requests.Session()

    def _post(self, endpoint: str, payload: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
        """Send a POST request to the vLLM server with retries.

        Args:
            endpoint: The API endpoint (e.g., "completions").
            payload: The JSON payload for the request.
            timeout: Request timeout in seconds.

        Returns:
            The JSON response from the server.

        Raises:
            RuntimeError: If the request fails after retries.
        """
        url = f"{self.base_url}/v1/{endpoint}"
        for attempt in range(3):
            try:
                # Log adapter-management requests at DEBUG to reduce noise.
                if endpoint in ["load_lora_adapter", "unload_lora_adapter"]:
                    logging.debug("vLLM Request attempt=%d endpoint=%s url=%s payload=%s", attempt + 1, endpoint, url, json.dumps(payload, default=str))
                resp = self.session.post(url, json=payload, timeout=timeout)
                # Log raw response status and the first 2KB at DEBUG.
                raw_preview = resp.text[:2048]
                overflow = '' if len(resp.text) <= 2048 else ' ...[truncated %d chars]' % (len(resp.text) - 2048)
                logging.debug(
                    "vLLM Raw Response endpoint=%s attempt=%d status=%d body=%s%s",
                    endpoint,
                    attempt + 1,
                    resp.status_code,
                    raw_preview.replace("\n", "\\n"),
                    overflow,
                )
                if resp.status_code == 200:
                    # Treat empty 200 responses as success for adapter ops.
                    if not resp.content:
                        if endpoint in ["load_lora_adapter", "unload_lora_adapter"]:
                            logging.debug("vLLM Response endpoint=%s status=200 empty-body treated-as-success", endpoint)
                        return {}
                    try:
                        return resp.json()
                    except ValueError:
                        # Treat non-JSON 200 responses as success for adapter ops.
                        if endpoint in ["load_lora_adapter", "unload_lora_adapter"]:
                            logging.debug("vLLM Response endpoint=%s status=200 non-json-body='%s' treated-as-success", endpoint, resp.text.strip())
                            return {}
                        raise
                
                # Log detailed error response.
                error_text = resp.text
                logging.warning("vLLM POST %s failed (status %d): %s", endpoint, resp.status_code, error_text)
                
                # Treat "already loaded" as success for load_lora_adapter.
                if endpoint == "load_lora_adapter" and resp.status_code == 400 and "already been loaded" in error_text:
                    logging.info("Adapter already loaded, proceeding.")
                    return {}

                resp.raise_for_status()
            except Exception as exc:
                logging.warning("vLLM POST %s attempt %d failed: %s", endpoint, attempt + 1, exc)
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"vLLM endpoint {endpoint} failed after retries")

    def generate(self, prompts: List[Union[str, List[Dict[str, str]]]], model: Optional[str], sampling: Dict[str, Any], stop_ids: List[int]) -> List[Dict[str, Any]]:
        """Generate completions for a list of prompts.

        Args:
            prompts: List of prompt strings or list of messages (if using chat API).
            model: The model name to use (optional, defaults to client's model).
            sampling: Dictionary of sampling parameters.
            stop_ids: List of token IDs to stop generation.

        Returns:
            A list of completion dictionaries.
        """
        if self.use_chat_api:
            # Use chat completions API; wrap string prompts as user messages.
            messages_list = []
            for p in prompts:
                if isinstance(p, str):
                    messages_list.append([{"role": "user", "content": p}])
                else:
                    messages_list.append(p)
            
            # Chat endpoints typically accept one message list per request, so we parallelize.
            choices = []
            
            def _call_chat(msgs):
                payload = {
                    "model": model or self.model_name,
                    "messages": msgs,
                    **sampling,
                    "stop_token_ids": stop_ids,
                }
                # Explicitly control thinking mode to avoid template defaults.
                payload["chat_template_kwargs"] = {"enable_thinking": self.thinking_mode}
                
                logging.debug("vLLM Chat Payload: %s", json.dumps(payload, default=str))
                res = self._post("chat/completions", payload, timeout=240)
                return res["choices"][0]  # Return the first choice.

            # Parallelize chat calls with threads.
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(messages_list), 32)) as executor:
                results = list(executor.map(_call_chat, messages_list))
            
            # Map chat responses to the completions-style {'text': ...} format.
            formatted_choices = []
            for r in results:
                content = r["message"]["content"]
                formatted_choices.append({"text": content})
            return formatted_choices

        else:
            # Use legacy completions API.
            payload = {
                "model": model or self.model_name,
                "prompt": prompts,
                **sampling,
                "stop_token_ids": stop_ids,
            }
            res = self._post("completions", payload, timeout=120 * max(1, len(prompts)))
            choices = res.get("choices")
            if not isinstance(choices, list):
                raise RuntimeError("Unexpected vLLM response: missing choices")
            return choices

    def load_adapter(self, adapter_name: str, adapter_path: Path) -> None:
        """Load a LoRA adapter into the vLLM server.

        Args:
            adapter_name: The name to assign to the loaded adapter.
            adapter_path: The path to the adapter on disk.
        """
        logging.debug("Loading LoRA adapter name=%s path=%s", adapter_name, adapter_path)
        self._post("load_lora_adapter", {"lora_name": adapter_name, "lora_path": str(adapter_path)})
        logging.debug("Load LoRA adapter request completed name=%s", adapter_name)

    def unload_adapter(self, adapter_name: str) -> None:
        """Unload a LoRA adapter from the vLLM server.

        Args:
            adapter_name: The name of the adapter to unload.
        """
        try:
            logging.debug("Unloading LoRA adapter name=%s", adapter_name)
            self._post("unload_lora_adapter", {"lora_name": adapter_name})
            logging.debug("Unload LoRA adapter request completed name=%s", adapter_name)
        except Exception as exc:
            logging.warning("Failed to unload adapter %s: %s", adapter_name, exc)


def accuracy_and_texts(
    client: VLLMClient,
    questions: List[Dict[str, str]],
    answer_model_ref: Optional[str],
    sampling: Dict[str, Any],
    stop_ids: List[int],
    instruct_model: bool,
    chain_of_thought: bool,
    grader: AnswerGrader,
) -> Dict[str, Any]:
    """Compute accuracy and retrieve generated texts for a set of questions.

    Args:
        client: The VLLMClient to use for generation.
        questions: List of question dictionaries.
        answer_model_ref: The model reference to use (e.g., adapter name).
        sampling: Sampling parameters.
        stop_ids: Stop token IDs.
        instruct_model: Whether to use instruction formatting.
        chain_of_thought: Whether to use chain-of-thought formatting.
        grader: The AnswerGrader instance.

    Returns:
        A dictionary containing accuracy, predictions, and verdicts.
    """
    prompts = format_answer_prompts(questions, instruct_model, chain_of_thought)
    logging.debug("Prompts generated: %s", json.dumps(prompts, indent=2))
    ans_out = client.generate(prompts, answer_model_ref, sampling, stop_ids)
    logging.debug("VLLM Output: %s", json.dumps(ans_out, indent=2))
    predictions = [chunk.get("text", "") for chunk in ans_out]
    if chain_of_thought:
        predictions = [extract_final_answer(p) for p in predictions]
    logging.debug("Predictions: %s", predictions)
    verdicts = grader.grade(questions, predictions)
    logging.debug("Verdicts: %s", verdicts)
    accuracy = sum(verdicts) / len(verdicts) if verdicts else 0.0
    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "verdicts": verdicts,
    }


def _setup_vllm_eval_worker(
    worker_id: int,
    gpu_id: int,
    vllm_host: str,
    port: int,
    worker_config: Dict[str, Any],
    run_log_dir: Path,
) -> Tuple[VLLMClient, AnswerGrader, Dict[str, Any], List[int], bool, bool, int]:
    """Shared initialisation for vLLM evaluation workers."""
    stage_name = worker_config.get("stage_name", "eval")
    log_path = run_log_dir / f"vllm_{stage_name}_worker_{worker_id}.log"
    sys.stdout = open(log_path, "w", buffering=1, encoding="utf-8")
    sys.stderr = sys.stdout

    setup_logging(
        worker_config.get("log_level", "INFO"),
        log_file=None,
        console=True,
    )

    import transformers

    transformers.logging.set_verbosity_info()
    transformers.logging.disable_default_handler()
    transformers.logging.enable_propagation()

    client = VLLMClient(
        f"http://{vllm_host}:{port}",
        worker_config["model_name"],
        use_chat_api=worker_config.get("instruct_model", False),
        thinking_mode=worker_config.get("thinking_mode", False),
    )

    grader = AnswerGrader(
        grader_type=worker_config["grader_type"],
        env_file=worker_config.get("env_file"),
    )

    sampling = worker_config["sampling"]
    stop_ids = worker_config["stop_ids"]
    instruct_model = worker_config["instruct_model"]
    chain_of_thought = worker_config["chain_of_thought"]

    logging.info("Eval Worker %d connected to vLLM at port %d", worker_id, port)

    return client, grader, sampling, stop_ids, instruct_model, chain_of_thought, port


def vllm_baseline_eval_worker(
    worker_id: int,
    gpu_id: int,
    vllm_host: str,
    port: int,
    result_queue: "mp.Queue[Dict[str, Any]]",
    progress_proxy: Any,
    progress_lock: Any,
    worker_config: Dict[str, Any],
    run_log_dir: Path,
    task_batch: List[Dict[str, Any]],
) -> None:
    """Worker dedicated to baseline evaluation tasks."""

    client, grader, sampling, stop_ids, instruct_model, chain_of_thought, _ = _setup_vllm_eval_worker(
        worker_id,
        gpu_id,
        vllm_host,
        port,
        worker_config,
        run_log_dir,
    )

    progress_lock = threading.Lock()

    if not task_batch:
        logging.info("Baseline Eval Worker %d received empty task batch; exiting", worker_id)
        return

    max_threads = max(1, min(int(worker_config.get("baseline_thread_pool", 8)), len(task_batch)))
    max_retries = max(1, int(worker_config.get("baseline_task_retries", 2)))

    def _process_task(task_payload: Dict[str, Any]) -> Dict[str, Any]:
        task_obj = BaselineEvalTask(**task_payload)
        self_edit_id = f"a{task_obj.article_index:03d}_t{task_obj.template_index:03d}_c{task_obj.completion_index:03d}"
        logging.info("Evaluating baseline for %s run %d", self_edit_id, task_obj.run_index)
        if task_obj.article_title:
            logging.debug("Article: %s", task_obj.article_title)
        logging.debug("Questions count: %d", len(task_obj.questions))

        update_progress(
            progress_proxy,
            self_edit_id,
            run_index=task_obj.run_index,
            lock=progress_lock,
            baseline_status="evaluating",
            owner=f"gpu{gpu_id}",
            baseline_start_time=time.time(),
        )

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                rep = accuracy_and_texts(
                    client,
                    task_obj.questions,
                    client.model_name,
                    sampling,
                    stop_ids,
                    instruct_model,
                    chain_of_thought,
                    grader,
                )

                update_progress(
                    progress_proxy,
                    self_edit_id,
                    run_index=task_obj.run_index,
                    lock=progress_lock,
                    baseline_status="done",
                    baseline_acc=f"{rep['accuracy']:.4f}",
                    baseline_end_time=time.time(),
                )

                return BaselineEvalResult(
                    article_index=task_obj.article_index,
                    template_index=task_obj.template_index,
                    completion_index=task_obj.completion_index,
                    accuracy=rep["accuracy"],
                    predictions=rep["predictions"],
                    verdicts=rep["verdicts"],
                    run_index=task_obj.run_index,
                ).model_dump()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logging.exception(
                    "Baseline Eval Worker %d self_edit %s run %d failed attempt %d/%d",
                    worker_id,
                    self_edit_id,
                    task_obj.run_index,
                    attempt,
                    max_retries,
                )
                time.sleep(min(3.0, 0.5 * attempt))

        update_progress(
            progress_proxy,
            self_edit_id,
            run_index=task_obj.run_index,
            lock=progress_lock,
            baseline_status="error",
            message=str(last_error),
            baseline_end_time=time.time(),
        )
        return {"status": "error", "error": str(last_error), "task": task_payload, "type": "baseline"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(_process_task, payload) for payload in task_batch]
        for future in concurrent.futures.as_completed(futures):
            try:
                result_queue.put(future.result())
            except Exception as exc:  # pragma: no cover - defensive catch
                logging.exception("Baseline Eval Worker %d encountered thread failure", worker_id)
                result_queue.put({"status": "error", "error": str(exc), "task": {}, "type": "baseline"})

    logging.info("Baseline Eval Worker %d exiting", worker_id)


def vllm_adapter_eval_worker(
    worker_id: int,
    gpu_id: int,
    vllm_host: str,
    port: int,
    task_queue: "mp.Queue[Optional[Dict[str, Any]]]",
    result_queue: "mp.Queue[Dict[str, Any]]",
    progress_proxy: Any,
    progress_lock: Any,
    worker_config: Dict[str, Any],
    run_log_dir: Path,
) -> None:
    """Worker dedicated to adapter evaluation tasks."""

    client, grader, sampling, stop_ids, instruct_model, chain_of_thought, _ = _setup_vllm_eval_worker(
        worker_id,
        gpu_id,
        vllm_host,
        port,
        worker_config,
        run_log_dir,
    )

    while True:
        try:
            task = task_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if task is None:
            break

        try:
            task_obj = AdapterEvalTask(**task)
            self_edit_id = task_obj.self_edit_id
            adapter_path = Path(task_obj.adapter_path)

            update_progress(
                progress_proxy,
                self_edit_id,
                run_index=task_obj.run_index,
                lock=progress_lock,
                adapter_status="evaluating",
                owner=f"gpu{gpu_id}",
                adapter_start_time=time.time(),
            )
            logging.info("Evaluating adapter %s run %d", self_edit_id, task_obj.run_index)
            if task_obj.article_title:
                logging.debug("Article: %s", task_obj.article_title)
            logging.debug("Questions count: %d", len(task_obj.questions))

            adapter_name = f"adapter_{self_edit_id}_{int(time.time() * 1000)}"

            if not adapter_path.exists():
                raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

            client.load_adapter(adapter_name, adapter_path)

            try:
                rep = accuracy_and_texts(
                    client,
                    task_obj.questions,
                    adapter_name,
                    sampling,
                    stop_ids,
                    instruct_model,
                    chain_of_thought,
                    grader,
                )

                result_queue.put(
                    AdapterEvalWorkerResult(
                        self_edit_id=self_edit_id,
                        accuracy=rep["accuracy"],
                        predictions=rep["predictions"],
                        verdicts=rep["verdicts"],
                        run_index=task_obj.run_index,
                    ).model_dump()
                )
                update_progress(
                    progress_proxy,
                    self_edit_id,
                    run_index=task_obj.run_index,
                    lock=progress_lock,
                    adapter_status="done",
                    adapter_acc=f"{rep['accuracy']:.4f}",
                    adapter_end_time=time.time(),
                )

            finally:
                client.unload_adapter(adapter_name)

        except Exception as exc:
            logging.exception("Adapter Eval Worker %d failed on task %s", worker_id, task)
            result_queue.put({"status": "error", "error": str(exc), "task": task, "type": "adapter"})
            if isinstance(task, dict) and "self_edit_id" in task:
                update_progress(
                    progress_proxy,
                    task["self_edit_id"],
                    run_index=task.get("run_index"),
                    lock=progress_lock,
                    adapter_status="error",
                    message=str(exc),
                    adapter_end_time=time.time(),
                )

    logging.info("Adapter Eval Worker %d exiting", worker_id)


class _ProgressHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for the progress server."""

    server_version = "SFTProgress/0.1"

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET requests."""
        state_fn: Callable[[], Dict[str, Any]] = getattr(self.server, "state_fn")  # type: ignore[attr-defined]
        run_log_dir: Path = getattr(self.server, "run_log_dir")  # type: ignore[attr-defined]
        
        if self.path.startswith("/logs/"):
            filename = self.path[len("/logs/"):]
            log_path = run_log_dir / filename
            # Basic guard against directory traversal.
            try:
                log_path = log_path.resolve()
                if not str(log_path).startswith(str(run_log_dir.resolve())):
                    self._respond(403, "text/plain", b"Access denied")
                    return
                if log_path.exists() and log_path.is_file():
                    with log_path.open("rb") as f:
                        content = f.read()
                    self._respond(200, "text/plain; charset=utf-8", content)
                else:
                    self._respond(404, "text/plain", b"Log file not found")
            except Exception as e:
                self._respond(500, "text/plain", str(e).encode("utf-8"))
            return

        state = state_fn()
        if self.path.startswith("/api/status"):
            payload = json.dumps(state, default=str).encode("utf-8")
            self._respond(200, "application/json", payload)
            return
        html = self._render_html(state, run_log_dir).encode("utf-8")
        self._respond(200, "text/html; charset=utf-8", html)

    def _respond(self, code: int, content_type: str, body: bytes) -> None:
        """Send an HTTP response.

        Args:
            code: HTTP status code.
            content_type: Content-Type header value.
            body: Response body bytes.
        """
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover
        """Log an arbitrary message.

        Args:
            fmt: Format string.
            *args: Arguments for the format string.
        """
        # Silence default HTTP request logging.
        pass

    @staticmethod
    def _render_html(state: Dict[str, Any], run_log_dir: Path) -> str:
        """Render the progress state as HTML (SPA Shell)."""
        return f"""
        <!DOCTYPE html>
        <html>
            <head>
                <title>SFT Progress</title>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; padding: 20px; background-color: #f5f5f5; }}
                    h2, h3 {{ color: #333; }}

                    /* Main Table Styles */
                    table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #e0e0e0; padding: 8px; vertical-align: top; text-align: left; }}
                    th {{ background: #f8f9fa; font-weight: 600; position: sticky; top: 0; z-index: 10; }}

                    /* GPU Table Styles */
                    .gpu-table {{ table-layout: auto; }}
                    .gpu-table td {{ vertical-align: top; }}

                    /* Article Cell */
                    .article-cell {{ width: 150px; background: #fafafa; font-size: 0.9em; }}

                    /* Grid inside cell */
                    .comp-grid {{ display: flex; flex-wrap: wrap; gap: 4px; }}
                    .comp-box {{
                        width: 40px; height: 40px;
                        display: flex; flex-direction: column; align-items: center; justify-content: center;
                        color: #333; font-size: 0.75em; font-weight: bold;
                        border: 1px solid #ccc; border-radius: 4px; cursor: pointer;
                        transition: transform 0.1s; background-color: #e0e0e0;
                        position: relative;
                        overflow: hidden;
                    }}
                    
                    /* Pills */
                    .pill {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin: 1px; }}
                    .pills-container {{ display: flex; flex-wrap: wrap; gap: 1px; }}
                    
                    /* Modal */
                    .modal {{ display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.4); }}
                    .modal-content {{ background-color: #fefefe; margin: 5% auto; padding: 20px; border: 1px solid #888; width: 80%; max-width: 1000px; border-radius: 8px; }}
                    .close {{ color: #aaa; float: right; font-size: 28px; font-weight: bold; cursor: pointer; }}
                    .close:hover, .close:focus {{ color: #000; }}
                    
                    /* Run Table in Modal */
                    .run-table {{ width: 100%; margin-top: 10px; }}
                    .run-table th {{ background: #f0f0f0; }}
                    
                    /* Progress Bar */
                    .progress-container {{
                        width: 100%; background-color: #f3f3f3; border-radius: 4px; height: 20px; position: relative; margin-top: 5px;
                    }}
                    .progress-bar {{
                        height: 100%; border-radius: 4px; background-color: #4CAF50; width: 0%;
                        transition: width 0.5s; text-align: center; line-height: 20px; color: white; font-size: 0.8em;
                    }}
                    .progress-text {{
                        position: absolute; width: 100%; text-align: center; line-height: 20px; font-size: 0.8em; color: #333;
                    }}
                </style>
                <script>
                    var refreshInterval = {PROGRESS_REFRESH_SECS * 1000};
                    var currentData = null;
                    var selectedSelfEditID = null;

                    function fetchData() {{
                        fetch('/api/status')
                            .then(response => response.json())
                            .then(data => {{
                                currentData = data;
                                render(data);
                            }})
                            .catch(err => console.error('Error fetching status:', err));
                    }}

                    function render(data) {{
                        document.getElementById('stage-header').innerText = 'Stage: ' + data.stage;
                        renderTimers(data.stage_timings);
                        renderTimers(data.stage_timings);
                        renderGpuStatus(data.gpu_status);
                        renderGrid(data);
                        renderLogs(data.logs);
                        if (selectedSelfEditID) {{
                            updateModal(selectedSelfEditID);
                        }}
                    }}

                    function renderTimers(timings) {{
                        if (!timings) return;
                        updateTimer('timer-training', 'Training', timings['training']);
                        updateTimer('timer-vllm-start', 'vLLM Start', timings['start_vllm']);
                        updateTimer('timer-baseline', 'Baseline', timings['baseline_eval']);
                        updateTimer('timer-adapter', 'Adapter', timings['adapter_eval']);
                    }}

                    function renderGpuStatus(gpuStatus) {{
                        if (!gpuStatus) return;
                        var container = document.getElementById('gpu-status-container');
                        if (!container) return;
                        
                        var html = '<table class="gpu-table"><thead><tr><th>GPU</th><th>Status</th></tr></thead><tbody>';
                        var gpuIds = Object.keys(gpuStatus).sort(function(a, b) {{ return parseInt(a) - parseInt(b); }});
                        
                        gpuIds.forEach(function(gpuId) {{
                            var statusList = gpuStatus[gpuId];
                            var statusStr = statusList.length > 0 ? statusList.join('<br>') : 'Idle';
                            html += '<tr><td>GPU ' + gpuId + '</td><td>' + statusStr + '</td></tr>';
                        }});
                        html += '</tbody></table>';
                        container.innerHTML = html;
                    }}

                    function updateTimer(elementId, label, timing) {{
                        var el = document.getElementById(elementId);
                        if (!timing || !timing.start) {{
                            el.innerText = label + ": -";
                            return; 
                        }}
                        var start = timing.start;
                        var end = timing.end;
                        var now = Date.now() / 1000;
                        var duration = end ? (end - start) : (now - start);
                        el.innerText = label + ": " + formatDuration(duration);
                        if (!end) {{
                            el.style.fontWeight = "bold";
                            el.style.color = "#2196F3";
                        }} else {{
                            el.style.fontWeight = "normal";
                            el.style.color = "inherit";
                        }}
                    }}

                    function formatDuration(seconds) {{
                        if (!seconds) return "-";
                        var m = Math.floor(seconds / 60);
                        var s = Math.floor(seconds % 60);
                        return m + "m " + s + "s";
                    }}

                    function renderGrid(data) {{
                        var headEl = document.getElementById('selfEdits-table-head');
                        var bodyEl = document.getElementById('selfEdits-table-body');
                        if (!headEl || !bodyEl) {{
                            return;
                        }}
                        var selfEdits = data.selfEdits;
                        var articles = {{}};
                        var allTemplates = new Set();

                        for (var SelfEditID in selfEdits) {{
                            var snapshot = selfEdits[SelfEditID];
                            var aIdx = snapshot.article !== undefined ? snapshot.article : -1;
                            var tIdx = snapshot.template !== undefined ? snapshot.template : -1;
                            var cIdx = snapshot.completion !== undefined ? snapshot.completion : -1;

                            if (!articles[aIdx]) {{
                                articles[aIdx] = {{
                                    title: snapshot.article_title || ("Article " + aIdx),
                                    templates: {{}}
                                }};
                            }}
                            if (!articles[aIdx].templates[tIdx]) {{
                                articles[aIdx].templates[tIdx] = [];
                            }}
                            articles[aIdx].templates[tIdx].push({{cIdx: cIdx, SelfEditID: SelfEditID, snapshot: snapshot}});
                            allTemplates.add(tIdx);
                        }}

                        var sortedTIndices = Array.from(allTemplates).sort(function(a, b) {{ return a - b; }});
                        var sortedAIndices = Object.keys(articles).sort(function(a, b) {{ return a - b; }});

                        // Header
                        var headerHtml = '<tr><th>Article</th>';
                        sortedTIndices.forEach(function(t) {{ headerHtml += '<th>Template ' + t + '</th>'; }});
                        headerHtml += '</tr>';
                        headEl.innerHTML = headerHtml;

                        // Body
                        var bodyHtml = '';
                        sortedAIndices.forEach(function(aIdx) {{
                            var article = articles[aIdx];
                            var title = article.title;
                            var shortTitle = title.length > 30 ? title.substring(0, 30) + '..' : title;

                            var rowHtml = '<tr><td class="article-cell" title="' + title.replace(/"/g, '&quot;') + '"><strong>A' + aIdx + '</strong><br><small>' + shortTitle + '</small></td>';

                            sortedTIndices.forEach(function(tIdx) {{
                                var completions = article.templates[tIdx] || [];
                                completions.sort(function(a, b) {{ return a.cIdx - b.cIdx; }});

                                var cellHtml = '<div class="comp-grid">';
                                completions.forEach(function(item) {{
                                    var snapshot = item.snapshot;
                                    var tStatus = (snapshot.status || "waiting").toLowerCase();
                                    var isDummy = snapshot.is_dummy === true;

                                    var boxStyle = "";
                                    var labelColor = "#333";

                                    // Training status coloring
                                    if (isDummy) {{
                                        if (tStatus.indexOf("dummy_complete") !== -1) {{
                                            boxStyle = "background-color: #4CAF50; color: white; border-color: #388E3C;";
                                            labelColor = "white";
                                        }} else {{
                                            boxStyle = "background-color: #e0e0e0; color: #757575; border-color: #ccc;";
                                            labelColor = "#757575";
                                        }}
                                    }} else if (tStatus.indexOf("running") !== -1 || tStatus.indexOf("training") !== -1) {{
                                        boxStyle = "background-color: #2196F3; color: white; border-color: #1976D2;";
                                        labelColor = "white";
                                    }} else if (tStatus.indexOf("error") !== -1 || tStatus.indexOf("failed") !== -1) {{
                                        boxStyle = "background-color: #F44336; color: white; border-color: #D32F2F;";
                                        labelColor = "white";
                                    }} else if (tStatus.indexOf("trained") !== -1 || tStatus.indexOf("eval") !== -1 || tStatus.indexOf("done") !== -1) {{
                                        boxStyle = "background-color: #fff; border-color: #4CAF50;";
                                    }} else {{
                                        boxStyle = "background-color: #e0e0e0; color: #999;";
                                    }}

                                    var pillsHtml = "";
                                    // Show pills if trained or evaluating
                                    if (isDummy || tStatus.indexOf("trained") !== -1 || tStatus.indexOf("eval") !== -1 || tStatus.indexOf("done") !== -1) {{
                                        var runs = snapshot.runs || {{}};
                                        var runKeys = Object.keys(runs).sort(function(a, b) {{ return parseInt(a) - parseInt(b); }});

                                        if (runKeys.length > 0) {{
                                            pillsHtml = '<div class="pills-container">';
                                            runKeys.forEach(function(k) {{
                                                var r = runs[k];
                                                var pColor = "#e0e0e0"; // Waiting

                                                var bStatus = r.baseline_status || "waiting";
                                                var aStatus = r.adapter_status || "waiting";

                                                if (bStatus.indexOf("error") !== -1 || aStatus.indexOf("error") !== -1) {{
                                                    pColor = "#F44336"; // Red
                                                }} else if (bStatus === "evaluating" || aStatus === "evaluating") {{
                                                    pColor = "#FF9800"; // Orange
                                                }} else if (bStatus === "done" && aStatus === "done") {{
                                                    pColor = "#4CAF50"; // Green
                                                }} else if (bStatus === "done") {{
                                                     pColor = "#2196F3"; // Blue
                                                }}

                                                pillsHtml += '<div class="pill" style="background-color: ' + pColor + ';" title="Run ' + k + '"></div>';
                                            }});
                                            pillsHtml += '</div>';
                                        }}
                                    }}

                                    // Calculate durations
                                    var now = Date.now() / 1000;
                                    
                                    // Training
                                    var tStart = snapshot.start_time;
                                    var tEnd = snapshot.end_time;
                                    var tDur = tEnd ? (tEnd - tStart) : (tStart ? (now - tStart) : 0);
                                    
                                    // Evaluation
                                    var runs = snapshot.runs || {{}};
                                    var runKeys = Object.keys(runs);
                                    var bTotal = 0;
                                    var aTotal = 0;
                                    
                                    runKeys.forEach(function(k) {{
                                        var r = runs[k];
                                        var bS = r.baseline_start_time;
                                        var bE = r.baseline_end_time;
                                        bTotal += bE ? (bE - bS) : (bS ? (now - bS) : 0);
                                        
                                        var aS = r.adapter_start_time;
                                        var aE = r.adapter_end_time;
                                        aTotal += aE ? (aE - aS) : (aS ? (now - aS) : 0);
                                    }});
                                    
                                    var tooltip = "Self Edit: " + item.SelfEditID + "\\\\n" +
                                                  "Train: " + formatDuration(tDur) + "\\\\n" +
                                                  "Baseline: " + formatDuration(bTotal) + "\\\\n" +
                                                  "Adapter: " + formatDuration(aTotal);
                                    if (isDummy) {{
                                        tooltip += "\\\\nDummy self_edit: baseline metrics reused";
                                    }}

                                    cellHtml += '<div class="comp-box" style="' + boxStyle + '" onclick="openModal(&quot;' + item.SelfEditID + '&quot;)" title="' + tooltip + '">';
                                    cellHtml += '<span style="color:' + labelColor + '">C' + item.cIdx + '</span>';
                                    cellHtml += pillsHtml;
                                    cellHtml += '</div>';
                                }});
                                cellHtml += '</div>';
                                rowHtml += '<td>' + cellHtml + '</td>';
                            }});
                            rowHtml += '</tr>';
                            bodyHtml += rowHtml;
                        }});
                        bodyEl.innerHTML = bodyHtml;
                    }}

                    function renderLogs(logs) {{
                        if (!logs || logs.length === 0) {{
                            document.getElementById('logs-list').innerHTML = '<p>No logs found.</p>';
                            return;
                        }}
                        var html = '';
                        logs.forEach(function(log) {{
                            html += '<li><a href="/logs/' + log + '" target="_blank">' + log + '</a></li>';
                        }});
                        document.getElementById('logs-list').innerHTML = html;
                    }}

                    function openModal(SelfEditID) {{
                        selectedSelfEditID = SelfEditID;
                        document.getElementById("infoModal").style.display = "block";
                        updateModal(SelfEditID);
                    }}

                    function updateModal(SelfEditID) {{
                        if (!currentData || !currentData.selfEdits[SelfEditID]) return;
                        var snapshot = currentData.selfEdits[SelfEditID];
                        var now = Date.now() / 1000;

                        var html = "<h3>Self Edit: " + SelfEditID + "</h3>";

                        // Training Info
                        var tStatus = snapshot.status || 'waiting';
                        var tStart = snapshot.start_time;
                        var tEnd = snapshot.end_time;
                        var tElapsed = tEnd ? (tEnd - tStart) : (tStart ? (now - tStart) : 0);

                        html += "<div style='display: flex; justify-content: space-between;'>";
                        html += "<div><strong>Training Status:</strong> " + tStatus + "</div>";
                        html += "<div><strong>GPU:</strong> " + (snapshot.owner || '-') + "</div>";
                        html += "<div><strong>Time:</strong> " + formatDuration(tElapsed) + "</div>";
                        html += "</div>";

                        if (tStatus === 'running') {{
                             html += '<div class="progress-container"><div class="progress-bar" style="width: 100%; animation: pulse 2s infinite;">Running...</div></div>';
                        }}

                        var runs = snapshot.runs || {{}};
                        var runKeys = Object.keys(runs).sort(function(a, b) {{ return parseInt(a) - parseInt(b); }});

                        if (runKeys.length > 0) {{
                            html += "<h4>Evaluation Runs</h4><table class='run-table'><thead><tr><th>Run</th><th>GPU</th><th>Baseline Status</th><th>Baseline Time</th><th>Baseline Acc</th><th>Adapter Status</th><th>Adapter Time</th><th>Adapter Acc</th></tr></thead><tbody>";
                            runKeys.forEach(function(k) {{
                                var r = runs[k];

                                var bStart = r.baseline_start_time;
                                var bEnd = r.baseline_end_time;
                                var bElapsed = bEnd ? (bEnd - bStart) : (bStart ? (now - bStart) : 0);

                                var aStart = r.adapter_start_time;
                                var aEnd = r.adapter_end_time;
                                var aElapsed = aEnd ? (aEnd - aStart) : (aStart ? (now - aStart) : 0);

                                html += "<tr>";
                                html += "<td>" + k + "</td>";
                                html += "<td>" + (r.owner || '-') + "</td>";
                                html += "<td>" + (r.baseline_status || 'waiting') + "</td>";
                                html += "<td>" + formatDuration(bElapsed) + "</td>";
                                html += "<td>" + (r.baseline_acc || '-') + "</td>";
                                html += "<td>" + (r.adapter_status || 'waiting') + "</td>";
                                html += "<td>" + formatDuration(aElapsed) + "</td>";
                                html += "<td>" + (r.adapter_acc || '-') + "</td>";
                                html += "</tr>";
                            }});
                            html += "</tbody></table>";
                        }} else {{
                            html += "<p><em>No evaluation runs yet.</em></p>";
                        }}
                        document.getElementById("modalBody").innerHTML = html;
                    }}

                    function closeModal() {{
                        document.getElementById("infoModal").style.display = "none";
                        selectedSelfEditID = null;
                    }}

                    window.onclick = function(event) {{
                        var modal = document.getElementById("infoModal");
                        if (event.target == modal) {{
                            closeModal();
                        }}
                    }}

                    window.onload = function() {{
                        fetchData();
                        setInterval(fetchData, refreshInterval);
                    }};
                </script>
            </head>
            <body>
                <h2 id="stage-header">Stage: Loading...</h2>
                <div id="global-timers" style="margin-bottom: 10px; font-size: 1.1em;">
                    <span id="timer-training">Training: -</span> | 
                    <span id="timer-vllm-start">vLLM Start: -</span> |
                    <span id="timer-baseline">Baseline: -</span> | 
                    <span id="timer-adapter">Adapter: -</span>
                </div>

                <div id="gpu-status-container" style="margin-bottom: 20px;"></div>

                <table id="selfEdits-table" class="selfEdits-table">
                    <thead id="selfEdits-table-head"></thead>
                    <tbody id="selfEdits-table-body"></tbody>
                </table>
                
                <div id="infoModal" class="modal">
                    <div class="modal-content">
                        <span class="close" onclick="closeModal()">&times;</span>
                        <div id="modalBody"></div>
                    </div>
                </div>
                
                <div class="logs">
                    <h3>Logs</h3>
                    <ul id="logs-list"></ul>
                </div>
            </body>
        </html>
        """.strip()


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Threaded HTTP server."""
    daemon_threads = True
    allow_reuse_address = True


class ProgressServer(threading.Thread):
    """Background thread running a simple HTTP progress server."""

    def __init__(self, state_fn: Callable[[], Dict[str, Any]], run_log_dir: Path, host: str = "127.0.0.1", port: int = 0) -> None:
        """Initialize the ProgressServer.

        Args:
            state_fn: Function that returns the current state dictionary.
            run_log_dir: Directory containing log files.
            host: Hostname to bind to.
            port: Port to bind to (0 for random).
        """
        super().__init__(daemon=True)
        self.state_fn = state_fn
        self.run_log_dir = run_log_dir
        self.host = host
        self.port = port
        self.httpd: Optional[ThreadedHTTPServer] = None

    def run(self) -> None:
        """Run the HTTP server."""
        handler = functools.partial(_ProgressHandler)
        server = ThreadedHTTPServer((self.host, self.port), handler)
        server.state_fn = self.state_fn  # type: ignore[attr-defined]
        server.run_log_dir = self.run_log_dir  # type: ignore[attr-defined]
        self.httpd = server
        self.port = server.server_address[1]
        logging.info("Progress server at http://%s:%d", self.host, self.port)
        logging.info("To access from local machine via VS Code, check the 'Ports' tab or run: ssh -L %d:localhost:%d user@remote_host", self.port, self.port)
        server.serve_forever()

    def stop(self) -> None:
        """Stop the HTTP server."""
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            logging.info("Progress server stopped")


def update_gpu_status(proxy: Any, gpu_id: int, status_list: List[str], lock: Optional[Any] = None) -> None:
    """Update the status list for a specific GPU.

    Args:
        proxy: The multiprocessing dict proxy for gpu_status.
        gpu_id: The ID of the GPU to update.
        status_list: List of status strings for the GPU.
        lock: Optional multiprocessing lock to synchronize updates.
    """
    def _update():
        proxy[gpu_id] = status_list

    if lock:
        with lock:
            _update()
    else:
        _update()

def update_progress(proxy: Any, self_edit_id: str, run_index: Optional[int] = None, lock: Optional[Any] = None, **fields: Any) -> None:
    """Update the progress status for a specific self_edit.

    Args:
        proxy: The multiprocessing dict proxy for progress tracking.
        self_edit_id: The ID of the self_edit to update.
        run_index: Optional index of the run to update.
        lock: Optional multiprocessing lock to synchronize updates.
        **fields: Key-value pairs to update in the self_edit's status.
    """
    def _update():
        snapshot = dict(proxy.get(self_edit_id, {}))
        if run_index is not None:
            runs = snapshot.get("runs", {})
            run_key = str(run_index)
            run_snapshot = runs.get(run_key, {})
            run_snapshot.update(fields)
            runs[run_key] = run_snapshot
            snapshot["runs"] = runs
        else:
            snapshot.update(fields)
        proxy[self_edit_id] = snapshot

    if lock:
        with lock:
            _update()
    else:
        _update()


def self_edit_executor(
    worker_id: int,
    gpu_id: int,
    task_queue: "mp.Queue[Optional[Dict[str, Any]]]",
    result_queue: "mp.Queue[Dict[str, Any]]",
    ready_event: Any,
    stop_event: Any,
    scratch_dir: Path,
    progress_proxy: Any,
    gpu_status_proxy: Any,
    progress_lock: Any,
    worker_config: Dict[str, Any],
    run_log_dir: Path,
) -> None:
    """Worker function to execute self edits on a GPU.

    Args:
        worker_id: Unique ID of the worker process.
        gpu_id: ID of the GPU to use.
        task_queue: Queue to receive tasks from.
        result_queue: Queue to send results to.
        ready_event: Event to signal when the worker is ready.
        stop_event: Event to signal when the worker should stop and shutdown.
        scratch_dir: Directory for temporary files.
        progress_proxy: Proxy for updating progress.
        worker_config: Configuration dictionary for the worker.
        run_log_dir: Directory for logs of this run.
    """
    # Redirect stdout/stderr to a per-worker log file (including HF Trainer output).
    log_path = run_log_dir / f"self_edit_executor_gpu{gpu_id}.log"
    sys.stdout = open(log_path, "w", buffering=1, encoding="utf-8")
    sys.stderr = sys.stdout

    # Configure logging to write to redirected stderr.
    setup_logging(
        worker_config.get("log_level", "INFO"),
        log_file=None,  # Rely on stderr redirection instead of a FileHandler.
        console=True,  # Add a StreamHandler to stderr.
    )

    # Route transformers logging through the root logger.
    import transformers
    transformers.logging.set_verbosity_info()
    transformers.logging.disable_default_handler()
    transformers.logging.enable_propagation()

    # Restrict this process to the assigned GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)

    # Set a per-worker random seed.
    random.seed(worker_id + gpu_id * 17)

    trainer = AdapterTrainer(
        model_name=worker_config["model_name"],
        instruct_model=worker_config.get("instruct_model", False),
        max_seq_length=worker_config.get("max_seq_length", DEFAULT_MAX_SEQ_LEN_FOR_TRAINING_ADAPTERS),
    )
    ready_event.set()
    logging.info("Worker %d on GPU %d ready", worker_id, gpu_id)

    while not stop_event.is_set():
        try:
            task = task_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if task is None:
            break
        self_edit_id = task["self_edit_id"]
        update_progress(progress_proxy, self_edit_id, lock=progress_lock, status="running", owner=f"gpu{gpu_id}", message="training", start_time=time.time())
        update_gpu_status(gpu_status_proxy, gpu_id, [f"Training {self_edit_id}"], lock=progress_lock)
        try:
            checkpoint = trainer.train(
                self_edit_id,
                task["training_sequences"],
                task["hyperparameters"],
                scratch_dir,
                gpu_id,
            )
            result_queue.put(
                {
                    "self_edit_id": self_edit_id,
                    "adapter_path": str(checkpoint),
                    "status": "success",
                    "article_index": task["article_index"],
                    "template_index": task["template_index"],
                    "completion_index": task["completion_index"],
                    "hyperparameters": task["hyperparameters"],
                    "training_sequence_count": len(task["training_sequences"]),
                }
            )
            update_progress(progress_proxy, self_edit_id, lock=progress_lock, status="trained", message="adapter saved", end_time=time.time())
            update_gpu_status(gpu_status_proxy, gpu_id, ["Idle"], lock=progress_lock)
        except Exception as exc:
            logging.exception("Worker %d failed on %s", worker_id, self_edit_id)
            result_queue.put({"self_edit_id": self_edit_id, "status": "error", "error": str(exc)})
            update_progress(progress_proxy, self_edit_id, lock=progress_lock, status="failed", message=str(exc), end_time=time.time())
            update_gpu_status(gpu_status_proxy, gpu_id, ["Idle (Error)"], lock=progress_lock)
    logging.info("Worker %d exiting", worker_id)


class SelfEditTask(BaseModel):
    """Represents a single self edit task to be executed."""
    self_edit_id: str
    article_index: int
    template_index: int
    completion_index: int
    article: Article
    hyperparameters: Hyperparameters
    training_sequences: List[str]
    has_training_data: bool = True


class AdapterRecord(BaseModel):
    """Record of a trained adapter."""
    self_edit_id: str
    article_index: int
    template_index: int
    completion_index: int
    checkpoint_path: Optional[Path] = None
    hyperparameters: Dict[str, Any]
    training_sequence_count: int
    is_dummy: bool = False


class AdapterEvalResult(BaseModel):
    """Result of evaluating an adapter."""
    self_edit_id: str
    article_index: int
    template_index: int
    completion_index: int
    adapter_accuracy: float
    adapter_std: float
    gain: float
    checkpoint_path: Optional[Path] = None
    question_details: List[List[Dict[str, Any]]]
    run_accuracies: List[float] = Field(default_factory=list)
    is_dummy: bool = False


# Represents one element of the (article, template, completion, run) 4D Cartesian Product.
class BaselineEvalTask(BaseModel):
    type: Literal["baseline"] = "baseline"
    article_index: int
    template_index: int
    completion_index: int
    questions: List[Dict[str, Any]]
    run_index: int = 0
    article_title: Optional[str] = None

# Represents one element of the (article, template, completion, run) 4D Cartesian Product.
# Note that the template index and completion index are absent because they're already fully determined by the adapter_path.
class AdapterEvalTask(BaseModel):
    type: Literal["adapter"] = "adapter"
    self_edit_id: str
    adapter_path: str
    article_index: int
    questions: List[Dict[str, Any]]
    run_index: int = 0
    article_title: Optional[str] = None


class BaselineEvalResult(BaseModel):
    status: Literal["success", "error"] = "success"
    type: Literal["baseline"] = "baseline"
    article_index: int
    template_index: int
    completion_index: int
    accuracy: float
    predictions: List[str]
    verdicts: List[Union[bool, int]]
    run_index: int
    error: Optional[str] = None


class AdapterEvalWorkerResult(BaseModel):
    status: Literal["success", "error"] = "success"
    type: Literal["adapter"] = "adapter"
    self_edit_id: str
    accuracy: float
    std: float = 0.0
    predictions: List[str]
    verdicts: List[Union[bool, int]]
    run_index: int
    error: Optional[str] = None


class Orchestrator:
    """Orchestrates the execution of self edits: training and evaluation."""

    def __init__(self, dataset: SelfEditDataset, args: argparse.Namespace) -> None:
        """Initialize the Orchestrator.

        Args:
            dataset: The dataset containing self edits.
            args: Parsed command-line arguments.
        """
        self.dataset = dataset
        self.args = args
        self.run_log_dir = args.run_log_dir
        self.num_gpus = args.num_gpus
        self.scratch_dir = ensure_dir(Path(args.scratch_dir))
        self.output_dir = ensure_dir(Path(args.output_dir))
        self.resume = bool(getattr(args, "resume", False))
        self.keep_scratch = bool(getattr(args, "keep_scratch", False))
        self.state_dir = self.scratch_dir / "state"
        self.adapter_registry_path = self.state_dir / "adapter_registry.json"
        self.stage_marker_path = self.state_dir / "stage_marker.json"
        self.baseline_runs_dir = self.state_dir / "baseline_runs"
        self.adapter_runs_dir = self.state_dir / "adapter_runs"
        self.run_successful = False

        # Queue of prepared `SelfEditTask` items to dispatch to workers.
        self.self_edit_tasks: List[SelfEditTask] = []

        # Registry of trained adapters keyed by `self_edit_id`.
        # Entries include checkpoint paths and metadata used during evaluation.
        self.adapter_registry: Dict[str, AdapterRecord] = {}
        if self.resume:
            ensure_dir(self.state_dir)
            self.stage_marker: Dict[str, Any] = self._read_json(self.stage_marker_path, {})
            self._load_adapter_registry_from_disk()
        else:
            if self.state_dir.exists():
                try:
                    shutil.rmtree(self.state_dir)
                except Exception as exc:  # pragma: no cover - defensive
                    logging.warning("Failed to clear previous state dir %s: %s", self.state_dir, exc)
            ensure_dir(self.state_dir)
            self.stage_marker = {}

        # Multiprocessing manager provides shared, proxy-backed state across processes.
        self.manager = mp.Manager()

        # Shared dictionary holding live state for the web dashboard.
        self.progress = self.manager.dict()  # type: ignore[var-annotated]

        # Per-self-edit status entries updated by workers and read by the web UI.
        self.progress["self_edits"] = self.manager.dict()

        # Current pipeline stage (e.g., "training", "baseline_eval", "adapter_eval").
        self.progress["stage"] = "initialising"
        self.progress["stage_timings"] = self.manager.dict()
        
        # GPU status strings keyed by GPU id.
        self.progress["gpu_status"] = self.manager.dict()
        for i in range(self.num_gpus):
            self.progress["gpu_status"][i] = []

        # Progress server thread reference for clean shutdown.
        self.progress_server: Optional[ProgressServer] = None

        # Lock for synchronizing progress updates across processes.
        self.progress_lock = self.manager.Lock()

        # vLLM server processes managed by this orchestrator.
        self.vllm_processes: List[subprocess.Popen] = []

    def run(self) -> None:
        """Run the full orchestration pipeline."""
        self.prepare_tasks()
        self._hydrate_progress_from_runs()
        self._sync_progress_from_registry()
        self.start_progress_server()
        try:
            self.progress["stage"] = "training"
            training_completed = self.resume and self._stage_completed("training")
            if training_completed:
                logging.info("Training stage already completed; reusing %d adapters", len(self.adapter_registry))
            else:
                self._update_stage_timing("training", start=True)
                self._set_stage_status("training", "in_progress")
                self.execute_training_stage()
                self._update_stage_timing("training", end=True)
                self._set_stage_status("training", "completed")

            # Start vLLM servers after training to free GPU memory.
            self._update_stage_timing("start_vllm", start=True)
            self.start_vllm_servers()
            self._update_stage_timing("start_vllm", end=True)

            self._update_stage_timing("baseline_eval", start=True)
            self.progress["stage"] = "baseline_eval"
            baseline_stats = self.run_baseline_evaluation()
            self._update_stage_timing("baseline_eval", end=True)
            self._set_stage_status("baseline_eval", "completed")

            self._update_stage_timing("adapter_eval", start=True)
            self.progress["stage"] = "adapter_eval"
            adapter_stats = self.run_adapter_evaluation(baseline_stats)
            self._update_stage_timing("adapter_eval", end=True)
            self._set_stage_status("adapter_eval", "completed")

            self.progress["stage"] = "writing_results"
            self.write_results(baseline_stats, adapter_stats)
            self._set_stage_status("writing_results", "completed")
            self.run_successful = True
        finally:
            self.stop_vllm_servers()
            self.stop_progress_server()
            if self.run_successful and not self.keep_scratch:
                self.cleanup_scratch()
            else:
                logging.info("Preserving scratch directory %s", self.scratch_dir)

    def _update_stage_timing(self, stage: str, start: bool = False, end: bool = False) -> None:
        """Update timing information for a stage."""
        timings = self.progress["stage_timings"]
        stage_data = timings.get(stage, {})
        if start:
            stage_data["start"] = time.time()
        if end:
            stage_data["end"] = time.time()
        timings[stage] = stage_data
        self.progress["stage_timings"] = timings

    def start_vllm_servers(self) -> None:
        """Start vLLM API servers on each GPU.
        
        To avoid race conditions during CUDA graph compilation (especially on cold start),
        we start GPU 0 first and wait for it to be fully ready before starting the others.
        """
        self.progress["stage"] = "starting_vllm"
        logging.info("Starting vLLM servers...")
        
        meta_args = self.dataset.metadata.get("args", {})
        model_name = meta_args.get("model", self.args.model_name)
        base_port = self.args.vllm_api_port_start
        eval_workers_per_gpu = max(1, self.args.eval_workers_per_gpu)
        target_fraction = 0.95 / eval_workers_per_gpu
        
        # Compute max LoRA rank from the dataset.
        max_lora_rank = 16
        for template in self.dataset.self_edit_templates:
            if template.hyperparameters.lora_rank > max_lora_rank:
                max_lora_rank = template.hyperparameters.lora_rank
        
        # Round up to the nearest power of two.
        max_lora_rank = 1 << (max_lora_rank - 1).bit_length()
        logging.info("Calculated max LoRA rank: %d", max_lora_rank)
        
        def launch_server(gpu_id: int, sub_id: int) -> subprocess.Popen:
            port = base_port + gpu_id * eval_workers_per_gpu + sub_id
            
            # All layers use the same target fraction based on eval_workers_per_gpu.
            vram_fraction = target_fraction
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = str("True")
            # Construct the vLLM launch command.
            cmd = [
                "vllm", "serve",
                model_name,
                "--host", self.args.vllm_host,
                "--port", str(port),
                "--gpu-memory-utilization", str(vram_fraction),
                "--trust-remote-code",
                "--enable-lora",
                "--max-lora-rank", str(max_lora_rank),
                "--max-model-len", str(self.args.eval_max_seq_length),
                "--disable-log-requests",
                "--enforce-eager",
            ]
            
            log_file = self.run_log_dir / f"vllm_server_gpu{gpu_id}_{sub_id}.log"
            fh = log_file.open("w")
            
            logging.info("Launching vLLM GPU %d (worker %d): %s", gpu_id, sub_id, " ".join(cmd))
            return subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)

        def wait_for_server(gpu_id: int, sub_id: int, timeout: int = 600) -> None:
            port = base_port + gpu_id * eval_workers_per_gpu + sub_id
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    # Check whether the server is responding.
                    resp = requests.get(f"http://{self.args.vllm_host}:{port}/health", timeout=1)
                    if resp.status_code == 200:
                        return
                except Exception:
                    pass
                
                # Check whether the process died.
                proc_idx = gpu_id * eval_workers_per_gpu + sub_id
                if proc_idx < len(self.vllm_processes):
                    proc = self.vllm_processes[proc_idx]
                    if proc.poll() is not None:
                        raise RuntimeError(f"vLLM server on GPU {gpu_id} (worker {sub_id}) died unexpectedly. Check logs in {self.run_log_dir}")
                
                time.sleep(2)
            raise RuntimeError(f"Timeout waiting for vLLM server on GPU {gpu_id} (worker {sub_id})")

        # Start vLLM servers layer-by-layer to reduce startup contention.
        logging.info("Initializing vLLM servers (staggered)...")
        
        for sub_id in range(eval_workers_per_gpu):
            logging.info("Starting vLLM Layer %d (Worker %d on all GPUs)...", sub_id, sub_id)
            
            # Layer 0: start GPU 0 first, then the rest.
            if sub_id == 0:
                # Start master on GPU 0.
                logging.info("Layer 0: Starting Master on GPU 0...")
                update_gpu_status(self.progress["gpu_status"], 0, [f"Starting vLLM Worker 0 ({target_fraction:.2f})"], lock=self.progress_lock)
                proc = launch_server(0, 0)
                self.vllm_processes.append(proc)
                
                # Wait for the master to become healthy.
                wait_for_server(0, 0)
                update_gpu_status(self.progress["gpu_status"], 0, ["vLLM Worker 0 Ready"], lock=self.progress_lock)
                logging.info("Layer 0: Master on GPU 0 is ready.")
                
                # Start remaining GPUs.
                logging.info("Layer 0: Starting remaining GPUs (1-%d)...", self.num_gpus - 1)
                for gpu_id in range(1, self.num_gpus):
                    update_gpu_status(self.progress["gpu_status"], gpu_id, [f"Starting vLLM Worker 0 ({target_fraction:.2f})"], lock=self.progress_lock)
                    proc = launch_server(gpu_id, 0)
                    self.vllm_processes.append(proc)
                
                # Wait for remaining GPUs to become healthy.
                for gpu_id in range(1, self.num_gpus):
                    wait_for_server(gpu_id, 0)
                    update_gpu_status(self.progress["gpu_status"], gpu_id, ["vLLM Worker 0 Ready"], lock=self.progress_lock)
                logging.info("Layer 0: All GPUs ready.")
                
            else:
                # For additional layers, start all GPUs in parallel after the prior layer is ready.
                
                logging.info("Layer %d: Starting all GPUs (VRAM: %.2f)...", sub_id, target_fraction)
                for gpu_id in range(self.num_gpus):
                    # Append a status entry for this worker.
                    current_status = self.progress["gpu_status"].get(gpu_id, [])
                    new_status = current_status + [f"Starting vLLM Worker {sub_id} ({target_fraction:.2f})"]
                    update_gpu_status(self.progress["gpu_status"], gpu_id, new_status, lock=self.progress_lock)
                    
                    proc = launch_server(gpu_id, sub_id)
                    self.vllm_processes.append(proc)
                
                # Wait for all GPUs in this layer.
                for gpu_id in range(self.num_gpus):
                    wait_for_server(gpu_id, sub_id)
                    
                    # Update status to ready.
                    current_status = self.progress["gpu_status"].get(gpu_id, [])
                    # Replace the last "Starting..." entry with "Ready".
                    if current_status and "Starting" in current_status[-1]:
                        current_status[-1] = f"vLLM Worker {sub_id} Ready"
                    else:
                        current_status.append(f"vLLM Worker {sub_id} Ready")
                    update_gpu_status(self.progress["gpu_status"], gpu_id, current_status, lock=self.progress_lock)
                
                logging.info("Layer %d: All GPUs ready.", sub_id)
            
        logging.info("All vLLM servers are ready.")

    def stop_vllm_servers(self) -> None:
        """Stop all vLLM servers."""
        if not self.vllm_processes:
            return
            
        logging.info("Stopping vLLM servers...")
        for proc in self.vllm_processes:
            if proc.poll() is None:
                proc.terminate()
        
        # Give processes a moment to shut down gracefully.
        time.sleep(2)
        
        for proc in self.vllm_processes:
            if proc.poll() is None:
                proc.kill()
        
        self.vllm_processes = []
        logging.info("All vLLM servers stopped.")

    def prepare_tasks(self) -> None:
        """Prepare self edit tasks from the dataset."""
        self_edits: List[SelfEditTask] = []
        for template_index, template in enumerate(self.dataset.self_edit_templates):
            for article_index, article in enumerate(self.dataset.articles):
                seq_sets = template.completions.get(str(article_index), [])
                for completion_index, completion in enumerate(seq_sets):
                    sequences = list(completion.training_sequences or [])
                    has_training = any((seq or "").strip() for seq in sequences)
                    # Example self_edit_id: a000_t001_c002.
                    self_edit_id = f"a{article_index:03d}_t{template_index:03d}_c{completion_index:03d}"
                    task = SelfEditTask(
                        self_edit_id=self_edit_id,
                        article_index=article_index,
                        template_index=template_index,
                        completion_index=completion_index,
                        article=article,
                        hyperparameters=template.hyperparameters,
                        training_sequences=sequences,
                        has_training_data=has_training,
                    )
                    self_edits.append(task)
                    self.progress["self_edits"][self_edit_id] = {
                        "status": "waiting",
                        "article": article_index,
                        "article_title": article.title,
                        "template": template_index,
                        "completion": completion_index,
                        "is_dummy": not has_training,
                        "runs": {str(r): {"status": "waiting"} for r in range(self.args.eval_times)}
                    }
        if not self_edits:
            raise RuntimeError("No self edit tasks were found in the dataset")
        self.self_edit_tasks = self_edits
        logging.info("Prepared %d self edit tasks", len(self_edits))

    def start_progress_server(self) -> None:
        """Start the progress monitoring server."""
        def snapshot() -> Dict[str, Any]:
            self_edits = {k: dict(v) for k, v in self.progress["self_edits"].items()}
            timings = {k: dict(v) for k, v in self.progress.get("stage_timings", {}).items()}
            gpu_status = {k: list(v) for k, v in self.progress.get("gpu_status", {}).items()}
            logs = [f.name for f in sorted(self.run_log_dir.glob("*.log"))] if self.run_log_dir.exists() else []
            return {"stage": self.progress.get("stage", "initialising"), "self_edits": self_edits, "logs": logs, "stage_timings": timings, "gpu_status": gpu_status}

        server = ProgressServer(snapshot, self.run_log_dir, host="0.0.0.0", port=self.args.progress_port)
        server.start()
        self.progress_server = server

    def stop_progress_server(self) -> None:
        """Stop the progress monitoring server."""
        if self.progress_server:
            self.progress_server.stop()

    def cleanup_scratch(self) -> None:
        """Clean up the scratch directory to free disk space."""
        if self.scratch_dir.exists():
            logging.info("Cleaning up scratch directory: %s", self.scratch_dir)
            try:
                shutil.rmtree(self.scratch_dir)
            except Exception as e:
                logging.warning("Failed to cleanup scratch directory: %s", e)

    # --- Persistence helpers -------------------------------------------------

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:  # pragma: no cover - best effort logging
            logging.warning("Failed to read JSON from %s: %s", path, exc)
            return default

    def _write_json_atomic(self, path: Path, payload: Any) -> None:
        ensure_dir(path.parent)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        tmp_path.replace(path)

    def _load_adapter_registry_from_disk(self) -> None:
        data = self._read_json(self.adapter_registry_path, {})
        for self_edit_id, record in data.items():
            checkpoint_str = record.get("checkpoint_path")
            checkpoint_path = Path(checkpoint_str) if checkpoint_str else None
            self.adapter_registry[self_edit_id] = AdapterRecord(
                self_edit_id=self_edit_id,
                article_index=record["article_index"],
                template_index=record["template_index"],
                completion_index=record["completion_index"],
                checkpoint_path=checkpoint_path,
                hyperparameters=record.get("hyperparameters", {}),
                training_sequence_count=record.get("training_sequence_count", 0),
                is_dummy=bool(record.get("is_dummy", False)),
            )

    def _persist_adapter_registry(self) -> None:
        payload = {
            self_edit_id: {
                "article_index": rec.article_index,
                "template_index": rec.template_index,
                "completion_index": rec.completion_index,
                "checkpoint_path": str(rec.checkpoint_path) if rec.checkpoint_path else None,
                "hyperparameters": rec.hyperparameters,
                "training_sequence_count": rec.training_sequence_count,
                "is_dummy": rec.is_dummy,
            }
            for self_edit_id, rec in self.adapter_registry.items()
        }
        self._write_json_atomic(self.adapter_registry_path, payload)

    def _set_stage_status(self, stage: str, status: str) -> None:
        self.stage_marker[stage] = {"status": status, "timestamp": time.time()}
        self._write_json_atomic(self.stage_marker_path, self.stage_marker)

    def _stage_completed(self, stage: str) -> bool:
        return self.stage_marker.get(stage, {}).get("status") == "completed"

    def _sync_progress_from_registry(self) -> None:
        if not self.adapter_registry:
            return
        for self_edit_id, record in self.adapter_registry.items():
            if self_edit_id not in self.progress["self_edits"]:
                continue
            snapshot = dict(self.progress["self_edits"].get(self_edit_id, {}))
            current_status = (snapshot.get("status") or "").lower()
            needs_status = current_status in ("", "waiting", "skipped_no_training")
            fields: Dict[str, Any] = {"is_dummy": record.is_dummy}
            if needs_status:
                fields["status"] = "dummy_waiting_baseline" if record.is_dummy else "trained"
            if not snapshot.get("message"):
                fields["message"] = "Resumed from snapshot"
            if fields:
                update_progress(
                    self.progress["self_edits"],
                    self_edit_id,
                    lock=self.progress_lock,
                    **fields,
                )

    def _self_edit_dir(self, stage: Literal["baseline", "adapter"], self_edit_id: str) -> Path:
        base = self.baseline_runs_dir if stage == "baseline" else self.adapter_runs_dir
        return ensure_dir(base / self_edit_id)

    def _persist_run(self, stage: Literal["baseline", "adapter"], self_edit_id: str, run_index: int, payload: Dict[str, Any]) -> None:
        self_edit_dir = self._self_edit_dir(stage, self_edit_id)
        target = self_edit_dir / f"run_{run_index}.json"
        tmp = target.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        tmp.replace(target)

    def _persist_baseline_run(self, result: Dict[str, Any]) -> None:
        self_edit_id = result.get("self_edit_id")
        if not self_edit_id:
            self_edit_id = f"a{result['article_index']:03d}_t{result['template_index']:03d}_c{result['completion_index']:03d}"
            result["self_edit_id"] = self_edit_id
        self._persist_run("baseline", self_edit_id, int(result.get("run_index", 0)), result)

    def _persist_adapter_run(self, result: Dict[str, Any]) -> None:
        self_edit_id = result.get("self_edit_id")
        if not self_edit_id:
            logging.warning("Adapter result missing self_edit_id; skipping persistence")
            return
        self._persist_run("adapter", self_edit_id, int(result.get("run_index", 0)), result)

    def _load_stage_runs(self, stage: Literal["baseline", "adapter"]) -> Dict[Tuple[str, int], Dict[str, Any]]:
        runs_dir = self.baseline_runs_dir if stage == "baseline" else self.adapter_runs_dir
        results: Dict[Tuple[str, int], Dict[str, Any]] = {}
        if not runs_dir.exists():
            return results
        for self_edit_dir in runs_dir.iterdir():
            if not self_edit_dir.is_dir():
                continue
            self_edit_id = self_edit_dir.name
            for run_file in self_edit_dir.glob("run_*.json"):
                try:
                    with run_file.open("r", encoding="utf-8") as fh:
                        data = json.load(fh)
                except Exception as exc:  # pragma: no cover - defensive
                    logging.warning("Failed to load %s: %s", run_file, exc)
                    continue
                run_idx = int(data.get("run_index", 0))
                data.setdefault("self_edit_id", self_edit_id)
                results[(self_edit_id, run_idx)] = data
        return results

    def _hydrate_stage_runs(self, stage: Literal["baseline", "adapter"]) -> Dict[str, List[float]]:
        runs = self._load_stage_runs(stage)
        self_edit_accs: Dict[str, Dict[int, float]] = {}
        status_key = "baseline_status" if stage == "baseline" else "adapter_status"
        acc_key = "baseline_acc" if stage == "baseline" else "adapter_acc"
        for (self_edit_id, run_idx), payload in runs.items():
            if self_edit_id not in self.progress["self_edits"]:
                continue
            status_value = payload.get("status", "success")
            fields: Dict[str, Any] = {}
            if status_value == "error":
                fields[status_key] = "error"
                if payload.get("error"):
                    fields["message"] = payload["error"]
            else:
                fields[status_key] = "done"
                acc = float(payload.get("accuracy", 0.0))
                fields[acc_key] = f"{acc:.4f}"
                self_edit_accs.setdefault(self_edit_id, {})[run_idx] = acc
            update_progress(
                self.progress["self_edits"],
                self_edit_id,
                run_index=run_idx,
                lock=self.progress_lock,
                **fields,
            )

        condensed: Dict[str, List[float]] = {}
        for self_edit_id, run_map in self_edit_accs.items():
            ordered = [run_map[idx] for idx in sorted(run_map.keys())]
            condensed[self_edit_id] = ordered
        return condensed

    def _mark_dummy_self_edit_complete(self, self_edit_id: str, run_accs: Sequence[float]) -> None:
        if self_edit_id not in self.progress["self_edits"]:
            return
        max_runs = min(len(run_accs), self.args.eval_times)
        if max_runs == 0:
            return
        now = time.time()
        for run_idx in range(max_runs):
            acc = run_accs[run_idx]
            update_progress(
                self.progress["self_edits"],
                self_edit_id,
                run_index=run_idx,
                lock=self.progress_lock,
                adapter_status="done",
                adapter_acc=f"{acc:.4f}",
                adapter_start_time=now,
                adapter_end_time=now,
            )
        update_progress(
            self.progress["self_edits"],
            self_edit_id,
            lock=self.progress_lock,
            status="dummy_complete",
            is_dummy=True,
            message="No training sequences; baseline metrics reused",
            end_time=now,
        )

    def _finalize_dummy_self_edits_from_baseline(
        self,
        baseline_stats: Dict[int, Dict[int, Dict[int, Dict[str, Any]]]],
    ) -> None:
        if not self.adapter_registry:
            return
        for self_edit_id, record in self.adapter_registry.items():
            if not record.is_dummy:
                continue
            a_stats = baseline_stats.get(record.article_index, {})
            t_stats = a_stats.get(record.template_index, {})
            c_stats = t_stats.get(record.completion_index)
            if not c_stats:
                continue
            runs = c_stats.get("runs", [])
            if len(runs) < self.args.eval_times:
                continue
            self._mark_dummy_self_edit_complete(self_edit_id, runs)

    def _hydrate_dummy_from_baseline(self, baseline_runs: Dict[str, List[float]]) -> None:
        if not baseline_runs:
            return
        for self_edit_id, run_accs in baseline_runs.items():
            if self_edit_id not in self.progress["self_edits"]:
                continue
            snapshot = dict(self.progress["self_edits"][self_edit_id])
            is_dummy = bool(snapshot.get("is_dummy"))
            if not is_dummy:
                record = self.adapter_registry.get(self_edit_id)
                is_dummy = bool(record and record.is_dummy)
            if not is_dummy:
                continue
            if len(run_accs) < self.args.eval_times:
                continue
            self._mark_dummy_self_edit_complete(self_edit_id, run_accs)

    def _hydrate_progress_from_runs(self) -> None:
        """Populate the live progress map with any persisted run metadata."""
        if "self_edits" not in self.progress:
            return
        baseline_runs = self._hydrate_stage_runs("baseline")
        self._hydrate_stage_runs("adapter")
        self._hydrate_dummy_from_baseline(baseline_runs)

    def execute_training_stage(self) -> None:
        """Execute the training stage using multiple GPUs."""
        existing_self_edit_ids = set(self.adapter_registry.keys())
        trainable_tasks: List[SelfEditTask] = []
        skipped_tasks: List[SelfEditTask] = []
        for task in self.self_edit_tasks:
            if not task.has_training_data:
                skipped_tasks.append(task)
            elif task.self_edit_id in existing_self_edit_ids:
                logging.info("Self edit %s already trained; skipping", task.self_edit_id)
            else:
                trainable_tasks.append(task)

        registry_dirty = False
        for task in skipped_tasks:
            self_edit_id = task.self_edit_id
            if self_edit_id in self.adapter_registry:
                continue
            self.adapter_registry[self_edit_id] = AdapterRecord(
                self_edit_id=self_edit_id,
                article_index=task.article_index,
                template_index=task.template_index,
                completion_index=task.completion_index,
                checkpoint_path=None,
                hyperparameters=task.hyperparameters.model_dump(),
                training_sequence_count=0,
                is_dummy=True,
            )
            registry_dirty = True
            update_progress(
                self.progress["self_edits"],
                self_edit_id,
                lock=self.progress_lock,
                status="dummy_waiting_baseline",
                is_dummy=True,
                message="No training sequences provided; will reuse baseline",
            )

        if registry_dirty:
            self._persist_adapter_registry()

        if not trainable_tasks:
            logging.info("No self edits contained training sequences; skipping training stage")
            return

        # Task queue for dispatching `SelfEditTask` messages to workers.
        # Size is bounded to avoid unbounded memory growth.
        task_queue: mp.Queue = mp.Queue(maxsize=self.num_gpus * 2)

        # Result queue for worker outcomes (success or failure).
        result_queue: mp.Queue = mp.Queue()

        # Stop event used as a shared termination signal.
        stop_event = mp.Event()

        workers: List[mp.Process] = []

        # Ready events: one per worker, set after initialization.
        ready_events: List[Any] = []

        meta_args = self.dataset.metadata.get("args", {})
        worker_config = {
            "model_name": meta_args.get("model", self.args.model_name),
            "instruct_model": bool(meta_args.get("instruct_model", False)),
            "max_seq_length": self.args.training_max_seq_length,
            "log_level": self.args.log_level,
        }

        # Spawn worker processes (one or more per GPU).
        executors_per_gpu = max(1, self.args.executors_per_gpu)
        for gpu_id in range(self.num_gpus):
            for i in range(executors_per_gpu):
                worker_id = gpu_id * executors_per_gpu + i
                ready = mp.Event()
                proc = mp.Process(
                    target=self_edit_executor,
                    args=(
                        # self_edit_executor expects both `worker_id` and `gpu_id`.
                        worker_id, 
                        gpu_id,
                    task_queue,
                    result_queue,
                    ready,
                    stop_event,
                    self.scratch_dir,
                    self.progress["self_edits"],
                    self.progress["gpu_status"],
                    self.progress_lock,
                    worker_config,
                    self.run_log_dir,
                ),
            )
            proc.daemon = True
            proc.start()
            workers.append(proc)
            ready_events.append(ready)

        # Wait for all workers to signal readiness before dispatching tasks.
        for ev in ready_events:
            ev.wait()
        
        logging.info("Dispatching %d tasks", len(trainable_tasks))
        
        # Producer-consumer pattern: orchestrator enqueues tasks, workers consume them.

        # Fan-out: enqueue all prepared tasks.
        for task in trainable_tasks:
            # Serialize tasks as plain dicts for cross-process transfer.
            msg = task.model_dump()
            
            # Enqueue unless the bounded queue is full.
            task_queue.put(msg)

        # Lookup map for retrying failed tasks.
        task_map = {t.self_edit_id: t for t in trainable_tasks}

        completed = 0
        total = len(trainable_tasks)

        # Collect results until all tasks succeed.
        while completed < total:
            result = result_queue.get()
            self_edit_id = result["self_edit_id"]

            if result["status"] == "success":
                # Success: record adapter path and metadata in the registry.
                record = AdapterRecord(
                    self_edit_id=self_edit_id,
                    article_index=result["article_index"],
                    template_index=result["template_index"],
                    completion_index=result["completion_index"],
                    checkpoint_path=Path(result["adapter_path"]),
                    hyperparameters=result["hyperparameters"],
                    training_sequence_count=result["training_sequence_count"],
                )
                self.adapter_registry[self_edit_id] = record
                self._persist_adapter_registry()
                completed += 1
            else:
                # Failure: log and re-queue for retry until all tasks succeed.
                error_msg = result.get("error", "Unknown error")
                logging.warning("Self edit %s failed during training: %s. Retrying...", self_edit_id, error_msg)
                
                # Retrieve the original task object.
                task = task_map[self_edit_id]
                
                # Re-serialize the task message.
                msg = task.model_dump()
                
                # Re-queue for the next available worker.
                task_queue.put(msg)

        # Shutdown sequence after all tasks complete.
        
        # Send one `None` sentinel per worker to stop them.
        for _ in workers:
            task_queue.put(None)
        
        # Also set the global stop_event as a broadcast stop signal.
        stop_event.set()
        
        # Wait for all worker processes to terminate.
        for proc in workers:
            proc.join()

    def _load_eval_tokenizer(self) -> Any:
        """Load the tokenizer for evaluation."""
        meta_args = self.dataset.metadata.get("args", {})
        model_name = meta_args.get("model", self.args.model_name)
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def _build_worker_config(self, stage_name: str) -> Dict[str, Any]:
        """Construct worker configuration shared between evaluation stages."""

        meta_args = self.dataset.metadata.get("args", {})
        instruct_model = bool(meta_args.get("instruct_model", False))
        tokenizer = self._load_eval_tokenizer()
        stop_token = "<|im_end|>" if instruct_model else (tokenizer.eos_token or "")

        return {
            "stage_name": stage_name,
            "model_name": meta_args.get("model", self.args.model_name),
            "instruct_model": instruct_model,
            "thinking_mode": bool(meta_args.get("thinking_mode", False)),
            "chain_of_thought": self.args.chain_of_thought,
            "grader_type": self.args.grader,
            "env_file": self.args.env_file,
            "log_level": self.args.log_level,
            "sampling": {
                "n": 1,
                "temperature": self.args.eval_temperature,
                "top_p": self.args.eval_top_p,
                "top_k": self.args.eval_top_k,
                "min_p": self.args.eval_min_p,
                "presence_penalty": self.args.eval_presence_penalty,
            },
            "stop_ids": tokenizer.encode(stop_token, add_special_tokens=False),
        }

    def _partition_baseline_tasks(self, tasks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        if self.num_gpus <= 0:
            raise ValueError("num_gpus must be positive for baseline evaluation")
        
        eval_workers_per_gpu = max(1, self.args.eval_workers_per_gpu)
        total_workers = self.num_gpus * eval_workers_per_gpu
        
        buckets: List[List[Dict[str, Any]]] = [[] for _ in range(total_workers)]
        for idx, task in enumerate(tasks):
            buckets[idx % total_workers].append(task)
        return buckets

    def _run_baseline_eval_stage(
        self,
        tasks: List[Dict[str, Any]],
        result_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Run baseline evaluation by assigning batched tasks per GPU."""

        if not tasks:
            return []

        worker_config = self._build_worker_config("baseline_eval")
        worker_config["baseline_thread_pool"] = max(1, getattr(self.args, "baseline_thread_pool", 4))
        worker_config["baseline_task_retries"] = max(1, getattr(self.args, "baseline_task_retries", 2))

        task_batches = self._partition_baseline_tasks(tasks)
        result_queue: mp.Queue = mp.Queue()
        workers: List[mp.Process] = []
        total = len(tasks)

        logging.info(
            "Starting %d batched baseline workers (tasks=%d)",
            sum(1 for batch in task_batches if batch),
            total,
        )

        eval_workers_per_gpu = max(1, self.args.eval_workers_per_gpu)
        for gpu_id in range(self.num_gpus):
            for sub_id in range(eval_workers_per_gpu):
                worker_idx = gpu_id * eval_workers_per_gpu + sub_id
                batch = task_batches[worker_idx]
                if not batch:
                    continue
                
                port = self.args.vllm_api_port_start + worker_idx
                
                proc = mp.Process(
                    target=vllm_baseline_eval_worker,
                    args=(
                        worker_idx,
                        gpu_id,
                        self.args.vllm_host,
                        port,
                        result_queue,
                        self.progress["self_edits"],
                        self.progress_lock,
                        worker_config,
                        self.run_log_dir,
                        batch,
                    ),
                )
                proc.daemon = True
                proc.start()
                workers.append(proc)

        if not workers:
            logging.warning("No baseline workers were started despite pending tasks")
            return []

        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        completed = 0

        while completed < total:
            res = result_queue.get()
            status = res.get("status")
            if status == "success":
                if result_callback:
                    result_callback(res)
                results.append(res)
            else:
                errors.append(res)
            completed += 1

        for proc in workers:
            proc.join()

        if errors:
            first = errors[0]
            failed_task = first.get("task", {})
            if isinstance(failed_task, dict) and {"article_index", "template_index", "completion_index"}.issubset(failed_task.keys()):
                self_edit_ref = (
                    f"a{failed_task['article_index']:03d}_t{failed_task['template_index']:03d}_c{failed_task['completion_index']:03d}"
                )
            else:
                self_edit_ref = "unknown-self-edit"
            raise RuntimeError(
                f"Baseline evaluation failed for {self_edit_ref}: {first.get('error', 'unknown error')}"
            )

        return results

    def _run_adapter_eval_stage(
        self,
        tasks: List[Dict[str, Any]],
        result_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Run adapter evaluation using the legacy queue-based fan-out."""

        if not tasks:
            return []

        task_queue: mp.Queue = mp.Queue()
        result_queue: mp.Queue = mp.Queue()

        task_map = {}
        for task in tasks:
            task_id = f"adpt_{task['self_edit_id']}_r{task['run_index']}"
            task["_task_id"] = task_id
            task_map[task_id] = task
            task_queue.put(task)

        eval_workers_per_gpu = max(1, self.args.eval_workers_per_gpu)
        total_workers = self.num_gpus * eval_workers_per_gpu

        worker_config = self._build_worker_config("adapter_eval")
        workers: List[mp.Process] = []

        logging.info("Starting %d workers for adapter_eval", total_workers)
        for gpu_id in range(self.num_gpus):
            for sub_id in range(eval_workers_per_gpu):
                worker_idx = gpu_id * eval_workers_per_gpu + sub_id
                port = self.args.vllm_api_port_start + worker_idx
                
                proc = mp.Process(
                    target=vllm_adapter_eval_worker,
                    args=(
                        worker_idx,
                        gpu_id,
                        self.args.vllm_host,
                        port,
                        task_queue,
                        result_queue,
                        self.progress["self_edits"],
                        self.progress_lock,
                        worker_config,
                        self.run_log_dir,
                    ),
                )
                proc.daemon = True
                proc.start()
                workers.append(proc)

        results: List[Dict[str, Any]] = []
        completed = 0
        total = len(tasks)

        while completed < total:
            res = result_queue.get()
            if res["status"] == "success":
                if result_callback:
                    result_callback(res)
                results.append(res)
                completed += 1
            elif res["status"] == "error":
                error_msg = res.get("error", "Unknown error")
                failed_task = res.get("task", {})
                task_id = f"adpt_{failed_task.get('self_edit_id')}_r{failed_task.get('run_index')}"

                logging.warning("Adapter eval task %s failed: %s. Retrying...", task_id, error_msg)

                if task_id in task_map:
                    task_queue.put(task_map[task_id])
                else:
                    logging.error("Could not find original adapter task for ID %s. Skipping retry.", task_id)
                    completed += 1

        for _ in range(total_workers):
            task_queue.put(None)

        for proc in workers:
            proc.join()

        return results

    def run_baseline_evaluation(self) -> Dict[int, Dict[int, Dict[int, Dict[str, Any]]]]:
        """Run baseline evaluation (no adapters).

        Returns:
            A dictionary of baseline statistics per article -> template -> completion.
        """
        logging.info("Preparing baseline evaluation tasks...")
        tasks: List[Dict[str, Any]] = []
        eval_times = self.args.eval_times
        existing_results = self._load_stage_runs("baseline")
        completed_keys = set(existing_results.keys())
        
        # We create one task per element of the Cartesian product (article, template, completion, run)
        for article_index, article in enumerate(self.dataset.articles):
            q_payload = [q.model_dump() for q in article.questions]
            for template_index in range(self.dataset.num_self_edit_templates):
                for completion_index in range(self.dataset.num_completions_per_template):
                    self_edit_id = f"a{article_index:03d}_t{template_index:03d}_c{completion_index:03d}"
                    for r in range(eval_times):
                        if (self_edit_id, r) in completed_keys:
                            continue
                        tasks.append(BaselineEvalTask(
                            article_index=article_index,
                            template_index=template_index,
                            completion_index=completion_index,
                            questions=q_payload,
                            run_index=r,
                            article_title=article.title
                        ).model_dump())
        if tasks:
            logging.info("Dispatching %d baseline tasks", len(tasks))
            new_results = self._run_baseline_eval_stage(tasks, result_callback=self._persist_baseline_run)
        else:
            logging.info("No new baseline tasks required; %d runs already present on disk", len(existing_results))
            new_results = []
        results = list(existing_results.values()) + new_results
        if not results:
            return {}
        
        # Aggregate results: stats[article][template][completion] = {mean, std, runs, details}
        stats: Dict[int, Dict[int, Dict[int, Dict[str, Any]]]] = {}
        
        # Group by (article, template, completion)
        grouped: Dict[tuple, List[Dict[str, Any]]] = {}
        for res in results:
            key = (res["article_index"], res["template_index"], res["completion_index"])
            grouped.setdefault(key, []).append(res)
            
        for (a_idx, t_idx, c_idx), res_list in grouped.items():
            res_list.sort(key=lambda r: r.get("run_index", 0))
            accs = [r["accuracy"] for r in res_list]
            q_details = []
            for r in res_list:
                q_details.append([
                    {
                        "question": q.question,
                        "answer": q.answer,
                        "prediction": pred,
                        "correct": verdict,
                        "run": r["run_index"],
                    }
                    for q, pred, verdict in zip(self.dataset.articles[a_idx].questions, r["predictions"], r["verdicts"])
                ])
            
            mean_acc = _stats.mean(accs)
            std_acc = _stats.stdev(accs) if len(accs) > 1 else 0.0
            
            if a_idx not in stats: stats[a_idx] = {}
            if t_idx not in stats[a_idx]: stats[a_idx][t_idx] = {}
            
            stats[a_idx][t_idx][c_idx] = {
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "runs": accs,
                "details": q_details,
            }
            
            # Update progress for the specific self_edit
            # We can now target the specific self_edit ID directly
            self_edit_id = f"a{a_idx:03d}_t{t_idx:03d}_c{c_idx:03d}"
            update_progress(self.progress["self_edits"], self_edit_id, baseline_acc=f"{mean_acc:.4f}", status="baseline_evaluated")

        self._finalize_dummy_self_edits_from_baseline(stats)
        return stats

    def run_adapter_evaluation(self, baseline_stats: Dict[int, Dict[int, Dict[int, Dict[str, Any]]]]) -> Dict[str, AdapterEvalResult]:
        """Run evaluation for all trained adapters.

        Args:
            baseline_stats: Baseline statistics to compute gains.

        Returns:
            A dictionary mapping self_edit IDs to evaluation results.
        """
        logging.info("Preparing adapter evaluation tasks...")
        tasks: List[Dict[str, Any]] = []
        eval_times = self.args.eval_times
        dummy_self_edits: List[str] = []
        existing_results = self._load_stage_runs("adapter")
        completed_keys = set(existing_results.keys())
        
        for self_edit_id, record in self.adapter_registry.items():
            if record.is_dummy or not record.checkpoint_path:
                dummy_self_edits.append(self_edit_id)
                continue
            article = self.dataset.articles[record.article_index]
            q_payload = [q.model_dump() for q in article.questions]
            for r in range(eval_times):
                if (self_edit_id, r) in completed_keys:
                    continue
                tasks.append(AdapterEvalTask(
                    self_edit_id=self_edit_id,
                    adapter_path=str(record.checkpoint_path),
                    article_index=record.article_index,
                    questions=q_payload,
                    run_index=r,
                    article_title=article.title
                ).model_dump())
            
        if tasks:
            logging.info("Dispatching %d adapter tasks", len(tasks))
            new_results = self._run_adapter_eval_stage(tasks, result_callback=self._persist_adapter_run)
        else:
            logging.info("No new adapter eval tasks required; %d runs already saved", len(existing_results))
            new_results = []
        results = list(existing_results.values()) + new_results
        
        # Aggregate results by self_edit_id
        grouped_results: Dict[str, List[Dict[str, Any]]] = {}
        for res in results:
            grouped_results.setdefault(res["self_edit_id"], []).append(res)
            
        adapter_results: Dict[str, AdapterEvalResult] = {}
        for self_edit_id, res_list in grouped_results.items():
            res_list.sort(key=lambda r: r.get("run_index", 0))
            record = self.adapter_registry[self_edit_id]
            per_run_accs = [float(r.get("accuracy", 0.0)) for r in res_list]
            
            mean_acc = _stats.mean(per_run_accs)
            std_acc = _stats.stdev(per_run_accs) if len(per_run_accs) > 1 else 0.0
            
            # Get baseline stats for this specific (article, template, completion)
            base_stats = baseline_stats.get(record.article_index, {}).get(record.template_index, {}).get(record.completion_index, {})
            base_mean = base_stats.get("mean_accuracy", 0.0)
            
            gain = mean_acc - base_mean
            
            # Collect details from all runs
            q_details = []
            for r in res_list:
                q_details.append([
                    {
                        "question": q.question,
                        "answer": q.answer,
                        "prediction": pred,
                        "correct": verdict,
                        "run": r["run_index"],
                    }
                    for q, pred, verdict in zip(self.dataset.articles[record.article_index].questions, r["predictions"], r["verdicts"])
                ])

            adapter_results[self_edit_id] = AdapterEvalResult(
                self_edit_id=self_edit_id,
                article_index=record.article_index,
                template_index=record.template_index,
                completion_index=record.completion_index,
                adapter_accuracy=mean_acc,
                adapter_std=std_acc,
                gain=gain,
                checkpoint_path=record.checkpoint_path,
                question_details=q_details,
                run_accuracies=per_run_accs,
                is_dummy=record.is_dummy,
            )
        
        # For dummy adapters (no training data), reuse baseline metrics directly.
        for self_edit_id in dummy_self_edits:
            record = self.adapter_registry[self_edit_id]
            base_stats = baseline_stats.get(record.article_index, {}).get(record.template_index, {}).get(record.completion_index, {})
            base_mean = base_stats.get("mean_accuracy", 0.0)
            base_std = base_stats.get("std_accuracy", 0.0)
            base_details = base_stats.get("details", [])
            base_runs = list(base_stats.get("runs", []))
            adapter_results[self_edit_id] = AdapterEvalResult(
                self_edit_id=self_edit_id,
                article_index=record.article_index,
                template_index=record.template_index,
                completion_index=record.completion_index,
                adapter_accuracy=base_mean,
                adapter_std=base_std,
                gain=0.0,
                checkpoint_path=record.checkpoint_path,
                question_details=base_details,
                run_accuracies=base_runs,
                is_dummy=True,
            )

        return adapter_results

    def write_results(self, baseline_stats: Dict[int, Dict[int, Dict[int, Dict[str, Any]]]], adapter_stats: Dict[str, AdapterEvalResult]) -> None:
        """Write evaluation results to disk following the specified JSON schema."""
        
        # Dimensions
        A = self.dataset.num_articles
        T = self.dataset.num_self_edit_templates
        C = self.dataset.num_completions_per_template
        E = self.args.eval_times
        
        # Initialize tensors with NaNs to surface missing runs.
        baseline_tensor = np.full((A, T, C, E), np.nan)
        adapter_tensor = np.full((A, T, C, E), np.nan)
        
        # Fill tensors: baseline.
        for a in range(A):
            for t in range(T):
                for c in range(C):
                    runs = baseline_stats.get(a, {}).get(t, {}).get(c, {}).get("runs", [])
                    for e in range(min(len(runs), E)):
                        baseline_tensor[a, t, c, e] = runs[e]
                        
        # Fill tensors: adapters (keyed by `self_edit_id`).
        for self_edit_id, res in adapter_stats.items():
            a = res.article_index
            t = res.template_index
            c = res.completion_index
            
            # Prefer persisted per-run accuracies (needed for resumed runs without question payloads).
            run_accs = list(res.run_accuracies)
            if not run_accs:
                # Fall back to recomputing from question details when available.
                for run_details in res.question_details:
                    if not run_details:
                        run_accs.append(0.0)
                        continue
                    correct_count = sum(1 for q in run_details if q.get("correct"))
                    run_accs.append(correct_count / len(run_details))
            
            for e in range(min(len(run_accs), E)):
                adapter_tensor[a, t, c, e] = run_accs[e]

        # Replace remaining NaNs (missing runs) with 0.0 for summary stats.
        if np.isnan(baseline_tensor).any():
            logging.warning("Some baseline results are missing (NaNs in tensor). Replacing with 0.0 for stats.")
            baseline_tensor = np.nan_to_num(baseline_tensor, nan=0.0)
        if np.isnan(adapter_tensor).any():
            logging.warning("Some adapter results are missing (NaNs in tensor). Replacing with 0.0 for stats.")
            adapter_tensor = np.nan_to_num(adapter_tensor, nan=0.0)

        # --- Overall metrics ---
        
        # Baseline metrics.
        baseline_mean_acc = float(np.mean(baseline_tensor))
        baseline_article_means = np.mean(baseline_tensor, axis=(1, 2, 3))  # Shape (A,)
        std_of_baseline_mean_acc_for_each_article = float(np.std(baseline_article_means))
        
        # Self-edit metrics.
        self_edit_mean_acc = float(np.mean(adapter_tensor))
        self_edit_article_means = np.mean(adapter_tensor, axis=(1, 2, 3))  # Shape (A,)
        std_of_self_edit_mean_acc_for_each_article = float(np.std(self_edit_article_means))
        
        # Noise metrics.
        # Evaluation noise: std over E (axis 3), then mean over A, T, C.
        self_edit_evaluation_noise = float(np.mean(np.std(adapter_tensor, axis=3)))
        
        # Instantiation noise: mean over E -> (A,T,C), std over C -> (A,T), mean over A,T.
        self_edit_template_instantiation_noise = float(np.mean(np.std(np.mean(adapter_tensor, axis=3), axis=2)))
        
        # Gain metrics.
        # Baseline article means: (A,).
        # Adapter means (A,T,C): mean over E.
        adapter_means_atc = np.mean(adapter_tensor, axis=3)
        # Broadcast baseline means to (A,T,C).
        baseline_means_broadcast = baseline_article_means[:, None, None]
        
        gains_atc = adapter_means_atc - baseline_means_broadcast
        self_edit_mean_gain = float(np.mean(gains_atc))
        
        wins_atc = adapter_means_atc > baseline_means_broadcast
        self_edit_win_rate = float(np.mean(wins_atc))

        remainder = 1 - baseline_means_broadcast
        self_edit_normalized_gain = float(np.mean(gains_atc / (remainder + 1e-6)))
        
        overall = {
            "baseline_mean_acc": baseline_mean_acc,
            "std_of_(baseline_mean_acc_for_each_article)": std_of_baseline_mean_acc_for_each_article,
            "self_edit_mean_acc": self_edit_mean_acc,
            "std_of_(self_edit_mean_acc_for_each_article)": std_of_self_edit_mean_acc_for_each_article,
            "self_edit_evaluation_noise": self_edit_evaluation_noise,
            "self_edit_template_instantiation_noise": self_edit_template_instantiation_noise,
            "self_edit_mean_gain": self_edit_mean_gain,
            "self_edit_win_rate": self_edit_win_rate,
            "self_edit_normalized_gain": self_edit_normalized_gain,
        }
        
        # --- Article statistics ---
        article_statistics = {}
        for a in range(A):
            # Slices.
            b_slice = baseline_tensor[a] # (T, C, E)
            a_slice = adapter_tensor[a]  # (T, C, E)
            b_mean_scalar = float(np.mean(b_slice))
            
            # Metrics.
            art_stats = {
                "baseline_mean_acc": b_mean_scalar,
                "std_of_(baseline_mean_acc_for_this_article)": float(np.std(b_slice)),
                "self_edit_mean_acc": float(np.mean(a_slice)),
                "std_of_(self_edit_mean_acc_for_this_article)": float(np.std(np.mean(a_slice, axis=(1, 2)))),  # Mean over C,E -> (T,); std over T.
                "self_edit_evaluation_noise": float(np.mean(np.std(a_slice, axis=2))),
                "self_edit_template_instantiation_noise": float(np.mean(np.std(np.mean(a_slice, axis=2), axis=1))),
                "self_edit_mean_gain": float(np.mean(np.mean(a_slice, axis=2) - b_mean_scalar)),
                "self_edit_win_rate": float(np.mean(np.mean(a_slice, axis=2) > b_mean_scalar)),
            }
            
            # Template statistics within each article.
            template_stats = {}
            for t in range(T):
                t_slice = a_slice[t]  # (C, E)
                
                t_stats = {
                    "self_edit_mean_acc": float(np.mean(t_slice)),
                    "self_edit_evaluation_noise": float(np.mean(np.std(t_slice, axis=1))),
                    "self_edit_template_instantiation_noise": float(np.std(np.mean(t_slice, axis=1))),
                    "self_edit_mean_gain": float(np.mean(np.mean(t_slice, axis=1) - b_mean_scalar)),
                    "self_edit_win_rate": float(np.mean(np.mean(t_slice, axis=1) > b_mean_scalar)),
                }
                
                # Completion-level stats.
                completions = {}
                for c in range(C):
                    c_slice = t_slice[c]  # (E,)
                    
                    # Resolve self-edit details for this completion.
                    self_edit_id = f"a{a:03d}_t{t:03d}_c{c:03d}"
                    adapter_res = adapter_stats.get(self_edit_id)
                    adapter_is_dummy = bool(adapter_res and adapter_res.is_dummy)
                    adapter_details = adapter_res.question_details if adapter_res else []

                    c_stats = {
                        "self_edit_mean_acc": float(np.mean(c_slice)),
                        "self_edit_evaluation_noise": float(np.std(c_slice)),
                        "self_edit_mean_gain": float(np.mean(c_slice - b_mean_scalar)),
                        "self_edit_win_rate": float(np.mean(c_slice > b_mean_scalar)),
                        "is_dummy_adapter": adapter_is_dummy,
                    }
                    
                    # Per-run evaluation details.
                    eval_runs = {}
                    # Reuse `adapter_details`/`adapter_res` resolved above.
                    baseline_res_runs = baseline_stats.get(a, {}).get(t, {}).get(c, {})
                    
                    for e in range(E):
                        b_acc = baseline_tensor[a, t, c, e]
                        a_acc = adapter_tensor[a, t, c, e]
                        
                        # Question-level details.
                        # Baseline details.
                        b_details = baseline_res_runs.get("details", [])[e] if baseline_res_runs and e < len(baseline_res_runs.get("details", [])) else []
                        # Adapter details.
                        a_details = adapter_details[e] if adapter_details and e < len(adapter_details) else []
                        
                        questions_dict = {}
                        # Assumes baseline and adapter questions are aligned.
                        for q_idx, (bq, aq) in enumerate(zip(b_details, a_details)):
                            questions_dict[str(q_idx)] = {
                                "baseline_answer": bq.get("prediction", ""),
                                "adapter_answer": aq.get("prediction", ""),
                                "baseline_correct": bq.get("correct", False),
                                "adapter_correct": aq.get("correct", False),
                            }
                            
                        eval_runs[str(e)] = {
                            "baseline_acc": float(b_acc),
                            "self_edit_acc": float(a_acc),
                            "gain": float(a_acc - b_acc),
                            "questions": questions_dict
                        }
                    
                    c_stats["evaluation_runs"] = eval_runs
                    completions[str(c)] = c_stats
                
                t_stats["completions"] = completions
                template_stats[str(t)] = t_stats
                
            art_stats["self_edit_template_statistics"] = template_stats
            article_statistics[str(a)] = art_stats

        # --- Global template statistics ---
        global_template_statistics = {}
        
        if self.args.archive_path:
            toolbox = []
            with open(self.args.archive_path, "r") as f:
                toolbox = json.load(f)
            logging.info("Loaded toolbox from %s", self.args.archive_path)
            logging.debug(f"Loaded toolbox looks like the following: {toolbox}")
        
        for t in range(T):
            # Slice across all articles for this template: (A, C, E).
            t_slice_global = adapter_tensor[:, t, :, :]
            
            # Metrics.
            # Mean accuracy.
            mean_acc = float(np.mean(t_slice_global))
            
            # Robustness across articles: mean over C,E -> (A,), std over A.
            robustness = float(np.std(np.mean(t_slice_global, axis=(1, 2))))
            
            # Evaluation noise: std over E -> (A,C), mean over A,C.
            eval_noise = float(np.mean(np.std(t_slice_global, axis=2)))
            
            # Instantiation noise: mean over E -> (A,C), std over C -> (A,), mean over A.
            inst_noise = float(np.mean(np.std(np.mean(t_slice_global, axis=2), axis=1)))
            
            # Gain.
            # Baseline means (A,), adapter means (A,C).
            a_means_ac = np.mean(t_slice_global, axis=2)
            gains = a_means_ac - baseline_article_means[:, None]
            mean_gain = float(np.mean(gains))
            
            # Win rate.
            wins = a_means_ac > baseline_article_means[:, None]
            win_rate = float(np.mean(wins))

            # Normalized gain.
            remainder = 1 - baseline_article_means[:, None]
            normalized_gain = float(np.mean(gains / remainder))
            
            global_template_statistics[str(t)] = {
                "self_edit_template_mean_acc": mean_acc,
                "std_of_(self_edit_template_mean_acc_across_articles)": robustness,
                "self_edit_template_evaluation_noise": eval_noise,
                "self_edit_template_instantiation_noise": inst_noise,
                "self_edit_template_mean_gain": mean_gain,
                "self_edit_template_win_rate": win_rate,
                "self_edit_template_normalized_gain": normalized_gain,
            }

            if self.args.archive_path:
                new_toolbox_entry = {
                    "data_creation_instruction": self.dataset.self_edit_templates[t].data_creation_instruction,
                    "hyperparameters": self.dataset.self_edit_templates[t].hyperparameters.model_dump(),
                    "accuracy": mean_acc,
                    "normalized_gain": normalized_gain
                }
                toolbox.append(new_toolbox_entry)
                logging.debug(f"Added the following to the toolbox: {new_toolbox_entry}")

        # Final payload.
        payload = {
            "overall": overall,
            "timings": dict(self.progress.get("stage_timings", {})),
            "timestamp": dt.datetime.now().isoformat(),
            "dataset_path": str(self.args.self_edit_data_path),
            "dataset": self.dataset.model_dump(),
            "exp_name": self.args.exp_name,
            "num_articles": A,
            "num_self_edit_templates": T,
            "num_completions_per_template": C,
            "eval_times": E,
            "article_statistics": article_statistics,
            "self_edit_template_statistics": global_template_statistics
        }
        
        # Write results to disk.
        output_file = self.output_dir / "results.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logging.info("Results written to %s", output_file)
        
        # Write adapter registry.
        registry_path = self.output_dir / "adapter_registry.json"
        with registry_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    self_edit_id: {
                        "article_index": rec.article_index,
                        "template_index": rec.template_index,
                        "completion_index": rec.completion_index,
                        "checkpoint_path": str(rec.checkpoint_path) if rec.checkpoint_path else None,
                        "is_dummy": rec.is_dummy,
                    }
                    for self_edit_id, rec in self.adapter_registry.items()
                },
                fh,
                indent=2,
            )
        logging.info("Wrote adapter registry to %s", registry_path)

        # Write toolbox if applicable.
        if self.args.archive_path:
            toolbox = sorted(toolbox, key=lambda x: x["accuracy"], reverse=True)
            toolbox_output_path = self.output_dir / "toolbox.json"
            with open(toolbox_output_path, "w") as f:
                json.dump(toolbox, f, indent=4)
            logging.info("Wrote toolbox to %s", toolbox_output_path)


def log_dataset_overview(dataset: SelfEditDataset) -> None:
    """Log a summary of the loaded dataset.

    Args:
        dataset: The SelfEditDataset to summarize.
    """
    num_articles = dataset.num_articles
    num_templates = dataset.num_self_edit_templates
    num_completions = dataset.num_completions_per_template
    total_self_edits = num_articles * num_templates * max(1, num_completions)
    sample_article = dataset.articles[0]
    sample_self_edit = dataset.self_edit_templates[0]
    sample_sequences: List[str] = []
    sample_completions = sample_self_edit.completions.get("0", [])
    if sample_completions:
        sample_sequences = list(sample_completions[0].training_sequences)
    
    logging.info(f"Args used to create the dataset: {json.dumps(dataset.metadata.get('args', {}), indent=2)}")
    logging.info(f"num_articles={num_articles}")
    logging.info(f"Example Article Title: {sample_article.title}")
    logging.info(f"num_self_edit_templates={num_templates}")
    logging.info(f"Example data creation instruction: {sample_self_edit.data_creation_instruction}")
    logging.info(f"num_completions_per_template={num_completions}")
    if sample_sequences:
        logging.info(f"Example training sequence: {sample_sequences[0]}")
    logging.info(f"Total Self Edits={total_self_edits}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: List of arguments (defaults to sys.argv).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Execute self edits in parallel")
    parser.add_argument("--self_edit_data_path", type=Path, required=True, help="Self edit dataset JSON file")
    parser.add_argument("--scratch_dir", type=Path, required=True, help="Directory for temporary files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for output results")
    parser.add_argument("--archive_path", type=Path, help="Directory for toolbox")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="Base model name or path. Note that if the SFT self edit data has a metadata field that specifies the model, that will overwrite whatever you pass here")
    parser.add_argument("--num_gpus", type=int, required=True, default=DEFAULT_NUM_GPUS)
    parser.add_argument("--training_max_seq_length", type=int, required=True, default=DEFAULT_MAX_SEQ_LEN_FOR_TRAINING_ADAPTERS, help="Maximum sequence length for training LoRA adapters. This is not used for the VLLM inference stage")
    parser.add_argument("--progress_port", type=int, default=0, help="Port for the HTTP server that shows progress of the training (put 0 for a random port)")
    parser.add_argument("--vllm_host", type=str, default="127.0.0.1", help="Hostname/IP for the vLLM API servers")
    parser.add_argument("--vllm_api_port_start", type=int, default=8001, help="Starting port number for the vLLM API servers (one per GPU)")
    parser.add_argument("--chain_of_thought", action="store_true", help="Put this flag if you want to use chain-of-thought prompting during evaluation. This is independent of the training process, only for evaluation")
    parser.add_argument("--eval_temperature", type=float, default=0.0, help="Temperature for evaluation sampling")
    parser.add_argument("--eval_top_p", type=float, default=1.0, help="Top-p for evaluation sampling")
    parser.add_argument("--eval_top_k", type=int, default=-1, help="Top-k for evaluation sampling")
    parser.add_argument("--eval_min_p", type=float, default=0.0, help="Min-p for evaluation sampling")
    parser.add_argument("--eval_presence_penalty", type=float, default=0.0, help="Presence penalty for evaluation sampling")
    parser.add_argument("--eval_max_seq_length", type=int, default=64, help="Maximum sequence length (prompt + completion) for evaluation")
    parser.add_argument("--eval_times", type=int, default=3, help="Number of evaluation runs per (article, template, completion) triplet")
    parser.add_argument("--executors_per_gpu", type=int, default=1, help="Number of executor processes per GPU")
    parser.add_argument("--eval_workers_per_gpu", type=int, default=2, help="Number of eval workers (and vLLM servers) per GPU")
    parser.add_argument("--grader", choices=["anthropic", "heuristic"], default="anthropic", help="Grading method to use")
    parser.add_argument("--env_file", type=Path, default=None, help="Path to .env file containing ANTHROPIC_API_KEY and secrets")
    parser.add_argument("--log_dir", type=Path, default=Path("logs"), help="Directory to store log files")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--exp_name", type=str, default="execute_self_edits", help="Experiment name to record in the results metadata")
    parser.add_argument("--resume", action="store_true", help="Resume from state stored inside --scratch_dir/state if available")
    parser.add_argument(
        "--keep-scratch",
        dest="keep_scratch",
        action="store_true",
        help="Preserve the scratch directory even after a successful run",
    )
    return parser.parse_args(argv)


def setup_logging(level: str, log_file: Optional[Path] = None, console: bool = True) -> None:
    """Configure the logging system.

    Args:
        level: Logging level string (e.g., "INFO", "DEBUG").
        log_file: Optional path to a log file.
        console: Whether to log to the console (stderr).
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Clear existing handlers to avoid duplication or unwanted inheritance
    for h in root.handlers[:]:
        root.removeHandler(h)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    if log_file:
        ensure_dir(log_file.parent)
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        root.addHandler(ch)

    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point for the script.

    Args:
        argv: Command-line arguments.
    """
    args = parse_args(argv)
    
    # Setup run-specific log directory
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = args.log_dir / f"run_{timestamp}"
    ensure_dir(run_log_dir)
    args.run_log_dir = run_log_dir

    # Update output_dir to include timestamp
    args.output_dir = args.output_dir / f"run_{timestamp}"
    ensure_dir(args.output_dir)
    
    # Configure orchestrator logging (console + file)
    setup_logging(args.log_level, run_log_dir / "orchestrator.log", console=True)
    
    dataset = SelfEditDataset.load(args.self_edit_data_path)
    log_dataset_overview(dataset)
    
    # Pass the run_log_dir to the orchestrator
    orchestrator = Orchestrator(dataset, args)
    orchestrator.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
        raise
