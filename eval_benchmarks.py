"""
Benchmark evaluation utilities with batch inference support.

Supports:
- MCQ-style datasets using loss-ranking over choices
- Generation-style datasets (e.g., GSM8K) with pass@k and avg@k
- HumanEval with pass@k metrics (code execution based)
"""

from __future__ import annotations

import json
import multiprocessing
import re
import signal
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from tqdm import tqdm


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    k: int = 1
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.5

    @property
    def do_sample(self) -> bool:
        return self.k > 1

    def to_generate_kwargs(self, pad_token_id: int) -> dict[str, Any]:
        """Convert to kwargs for model.generate()."""
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": pad_token_id,
            "num_return_sequences": self.k,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            kwargs.update({
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
            })
        return kwargs


# ============================================================================
# Code Execution (HumanEval)
# ============================================================================

def execute_code(
    code: str,
    test: str,
    entry_point: str,
    timeout: float = 5.0,
    use_multiprocess: bool = False,
) -> tuple[bool, str]:
    """Execute code with test cases. Returns (passed, result)."""
    full_code = f"{code}\n{test}\ncheck({entry_point})"

    def run() -> tuple[bool, str]:
        try:
            exec(full_code, {})
            return True, "passed"
        except AssertionError as e:
            return False, f"assertion_error: {e}"
        except Exception as e:
            return False, f"error: {type(e).__name__}: {e}"

    if use_multiprocess:
        return _execute_with_multiprocess(run, timeout)
    return _execute_with_signal(run, full_code, timeout)


def _execute_with_multiprocess(run_fn: Callable, timeout: float) -> tuple[bool, str]:
    """Execute with multiprocessing for isolation."""
    try:
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()

        def worker(q):
            q.put(run_fn())

        proc = ctx.Process(target=worker, args=(queue,))
        proc.start()
        proc.join(timeout=timeout)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            return False, "timed_out"

        return queue.get() if not queue.empty() else (False, "no_result")
    except Exception as e:
        return False, f"execution_error: {e}"


def _execute_with_signal(
    run_fn: Callable, full_code: str, timeout: float
) -> tuple[bool, str]:
    """Execute with signal-based timeout (Unix only, faster)."""
    if not hasattr(signal, "SIGALRM"):
        return run_fn()

    def handler(signum, frame):
        raise TimeoutError()

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(timeout))
    try:
        exec(full_code, {})
        signal.alarm(0)
        return True, "passed"
    except TimeoutError:
        return False, "timed_out"
    except AssertionError as e:
        return False, f"assertion_error: {e}"
    except Exception as e:
        return False, f"error: {type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ============================================================================
# Answer Extraction & Matching
# ============================================================================

def extract_gsm8k_answer(text: str | None) -> str | None:
    """Extract answer after '####' delimiter."""
    if not text or "####" not in text:
        return None
    return text.split("####")[-1].strip()


def extract_number(text: str | None) -> str | None:
    """Extract the last number from text."""
    if not text:
        return None
    matches = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return matches[-1] if matches else None


def check_answer(pred: str, gold: str | None) -> bool:
    """Check if prediction matches gold (numeric or exact)."""
    if gold is None:
        return False
    pred_num, gold_num = extract_number(pred), extract_number(gold)
    if pred_num and gold_num:
        try:
            return float(pred_num) == float(gold_num)
        except ValueError:
            pass
    return pred.strip() == gold.strip()


# ============================================================================
# Data Loading
# ============================================================================

def load_jsonl(path: str, limit: int | None = None) -> list[dict]:
    """Load JSONL file with optional limit."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
                if limit and len(items) >= limit:
                    break
    return items


def extract_prompt_and_gold(item: dict, tokenizer) -> tuple[str, str | None]:
    """Extract prompt and gold answer from item."""
    if "messages" in item:
        msgs = [m for m in item["messages"] if m["role"] != "assistant"]
    elif "question" in item:
        msgs = [{"role": "user", "content": item["question"]}]
    else:
        msgs = [{"role": "user", "content": item.get("prompt", "")}]

    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    gold = None
    if "answer" in item:
        gold = item["answer"]
    elif "answers" in item and item["answers"]:
        gold = item["answers"][0]
    elif "solution" in item:
        gold = extract_gsm8k_answer(item["solution"])
    elif "messages" in item:
        for m in item["messages"]:
            if m.get("role") == "assistant":
                content = m.get("content", "")
                gold = extract_gsm8k_answer(content) or content.strip()
                break

    return prompt, gold


# ============================================================================
# MCQ Evaluation (Batch)
# ============================================================================

def evaluate_mcq(
    model,
    tokenizer,
    items: list[dict],
    device: str = "cuda",
    batch_size: int = 8,
    top_ns: list[int] | None = None,
    fewshot_items: list[dict] | None = None,
) -> dict[str, Any]:
    """Evaluate MCQ by ranking choices with loss (batch inference)."""
    top_ns = sorted(set(top_ns or [1]))
    model.eval()

    correct = {n: 0 for n in top_ns}
    total = 0
    details = []

    # Prepare few-shot context
    fewshot_ctx = []
    if fewshot_items:
        letters = "ABCDEFGHIJ"
        for fs in fewshot_items:
            fs_q = fs.get("question") or fs.get("prompt")
            if "messages" in fs:
                # Use messages content if available
                user_msgs = [m["content"] for m in fs["messages"]
                             if m["role"] == "user"]
                fs_q = user_msgs[0]

            fs_choices = fs.get("choices") or fs.get("options") or []
            fs_label = fs.get("label") or fs.get("answer_key")
            
            if fs_q and fs_choices and fs_label is not None:
                content = f"{fs_q}"
                answer_content = f"{letters[fs_label]}. {fs_choices[fs_label]}"
                fewshot_ctx.extend([
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": answer_content}
                ])

    # Process items in batches
    num_items = len(items)
    num_batches = (num_items + batch_size - 1) // batch_size
    for batch_start in tqdm(
        range(0, num_items, batch_size), desc="MCQ", total=num_batches
    ):
        batch_items = items[batch_start:batch_start + batch_size]

        for item in batch_items:
            if "messages" in item:
                ctx = [m for m in item["messages"] if m["role"] != "assistant"]
            elif "question" in item:
                ctx = [{"role": "user", "content": item["question"]}]
            else:
                ctx = [{"role": "user", "content": item.get("prompt", "")}]

            # Prepend few-shot context
            full_ctx = fewshot_ctx + ctx

            choices = item.get("choices") or item.get("options") or []
            label = item.get("label") or item.get("answer_key")
            if not choices or label is None:
                continue

            letters = "ABCDEFGHIJ"
            texts = [
                tokenizer.apply_chat_template(
                    full_ctx + [{"role": "assistant", "content": f"{letters[i]}. {c}"}],
                    tokenize=False,
                )
                for i, c in enumerate(choices)
            ]

            enc = tokenizer(
                texts, padding=True, truncation=True,
                max_length=2048, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                logits = model(input_ids=enc.input_ids).logits[:, :-1]
                labels_shifted = enc.input_ids[:, 1:]
                loss_fn = nn.CrossEntropyLoss(reduction="none")
                losses = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    labels_shifted.reshape(-1),
                ).view(len(choices), -1)

                avg_losses = []
                for i in range(len(choices)):
                    m = labels_shifted[i] != tokenizer.pad_token_id
                    avg_losses.append(losses[i][m].mean().item())

            ranked = sorted(range(len(avg_losses)), key=lambda x: avg_losses[x])
            pred = ranked[0]
            total += 1

            for n in top_ns:
                if label in ranked[:n]:
                    correct[n] += 1

            details.append({
                "choices": choices,
                "gold": label,
                "pred": pred,
                "correct": pred == label,
            })

    results = {"total": total, "details": details}
    for n in top_ns:
        results[f"accuracy@{n}"] = correct[n] / max(total, 1)
    results["accuracy"] = results.get("accuracy@1", 0.0)
    return results


# ============================================================================
# Generation Evaluation (Batch)
# ============================================================================

def evaluate_generation(
    model,
    tokenizer,
    items: list[dict],
    config: GenerationConfig,
    device: str = "cuda",
    batch_size: int = 4,
    fewshot_items: list[dict] | None = None,
) -> dict[str, Any]:
    """Evaluate generation tasks with batch inference."""
    model.eval()
    gen_kwargs = config.to_generate_kwargs(tokenizer.pad_token_id)
    results = []

    # Prepare few-shot context
    fewshot_ctx = []
    if fewshot_items:
        for fs in fewshot_items:
            # Extract basic prompt and gold
            if "messages" in fs:
                msgs = [m for m in fs["messages"] if m["role"] != "assistant"]
                user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
                fs_q = user_msgs[0]
                # Find gold from assistant message
                fs_a = ""
                for m in fs["messages"]:
                    if m["role"] == "assistant":
                        fs_a = m["content"]
                        break
            else:
                fs_q = fs.get("question") or fs.get("prompt")
                fs_a = fs.get("answer") or fs.get("solution")
                if not fs_a and fs.get("answers"):
                    fs_a = fs.get("answers")[0]
            
            if fs_q and fs_a:
                fewshot_ctx.extend([
                    {"role": "user", "content": fs_q},
                    {"role": "assistant", "content": fs_a}
                ])

    # Prepare all prompts and golds
    all_prompts, all_golds = [], []
    for item in items:
        if "messages" in item:
            msgs = [m for m in item["messages"] if m["role"] != "assistant"]
        elif "question" in item:
            msgs = [{"role": "user", "content": item["question"]}]
        else:
            msgs = [{"role": "user", "content": item.get("prompt", "")}]

        # Prepend few-shot context
        full_msgs = fewshot_ctx + msgs
        
        prompt = tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        
        # Get gold answer
        _, gold = extract_prompt_and_gold(item, tokenizer)
        
        all_prompts.append(prompt)
        all_golds.append(gold)

    # Batch inference
    num_batches = (len(all_prompts) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="Generation"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(all_prompts))
        batch_prompts = all_prompts[start:end]
        batch_golds = all_golds[start:end]
        # Tokenize batch (padding_side should be "left" for generation)
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            # Output shape: (actual_batch_size * k, seq_len)
            outputs = model.generate(**enc, **gen_kwargs)

        # Get input lengths for each item (accounting for left padding)
        input_lens = enc.attention_mask.sum(dim=1).tolist()

        # Process each item in batch
        for i, gold in enumerate(batch_golds):
            input_len = input_lens[i]
            completions = []

            for j in range(config.k):
                # Index into flattened outputs: batch_size * k
                idx = i * config.k + j
                text = tokenizer.decode(
                    outputs[idx, input_len:], skip_special_tokens=True
                )
                pred = extract_gsm8k_answer(text) if "####" in text else text
                is_correct = check_answer(pred, gold) if gold else False
                completions.append({
                    "text": text,
                    "pred": extract_number(pred) or pred,
                    "correct": is_correct,
                })

            n_correct = sum(c["correct"] for c in completions)
            results.append({
                "prompt": batch_prompts[i],
                "gold": gold,
                "completions": completions,
                "first_correct": completions[0]["correct"],
                "any_correct": n_correct > 0,
                "avg_correct": n_correct / config.k,
            })

    return _aggregate_results(results, config.k)


# ============================================================================
# HumanEval Evaluation (Batch Generation + Sequential Execution)
# ============================================================================

def evaluate_humaneval(
    model,
    tokenizer,
    items: list[dict],
    config: GenerationConfig,
    device: str = "cuda",
    batch_size: int = 4,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Evaluate HumanEval with batch generation (execution is sequential)."""
    model.eval()
    gen_kwargs = config.to_generate_kwargs(tokenizer.pad_token_id)
    results = []
    stop_seqs = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint("]

    # Prepare all prompts
    all_chat_prompts = []
    all_items_data = []  # (task_id, prompt, test, entry_point)
    for item in items:
        task_id = item.get("task_id", "unknown")
        prompt, test, entry = item["prompt"], item["test"], item["entry_point"]
        msgs = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        all_chat_prompts.append(chat_prompt)
        all_items_data.append((task_id, prompt, test, entry))

    # Batch generation
    all_completions = []  # List of lists: [[k completions for item 0], ...]
    num_batches = (len(all_chat_prompts) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="HumanEval Gen"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(all_chat_prompts))
        batch_prompts = all_chat_prompts[start:end]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**enc, **gen_kwargs)

        input_lens = enc.attention_mask.sum(dim=1).tolist()

        for i in range(len(batch_prompts)):
            input_len = input_lens[i]
            item_completions = []
            for j in range(config.k):
                idx = i * config.k + j
                completion = tokenizer.decode(
                    outputs[idx, input_len:], skip_special_tokens=True
                )
                # Truncate at stop sequences
                for stop in stop_seqs:
                    if stop in completion:
                        completion = completion[:completion.index(stop)]
                item_completions.append(completion)
            all_completions.append(item_completions)

    # Sequential execution (cannot be batched)
    for i, (task_id, prompt, test, entry) in enumerate(
        tqdm(all_items_data, desc="HumanEval Exec")
    ):
        completions = []
        for completion in all_completions[i]:
            passed, result = execute_code(
                prompt + completion, test, entry, timeout=timeout
            )
            completions.append({
                "completion": completion,
                "passed": passed,
                "result": result,
            })

        n_passed = sum(c["passed"] for c in completions)
        results.append({
            "task_id": task_id,
            "completions": completions,
            "first_correct": completions[0]["passed"],
            "any_correct": n_passed > 0,
            "avg_correct": n_passed / config.k,
        })

    out = _aggregate_results(results, config.k)
    out["tasks_with_any_correct"] = sum(r["any_correct"] for r in results)
    return out


def _aggregate_results(results: list[dict], k: int) -> dict[str, Any]:
    """Aggregate per-item results into final metrics."""
    n = len(results)
    if n == 0:
        return {"num_tasks": 0, "k": k, "pass@1": 0.0, "accuracy": 0.0, "details": []}

    pass_1 = sum(r["first_correct"] for r in results) / n
    pass_k = sum(r["any_correct"] for r in results) / n
    avg_k = sum(r["avg_correct"] for r in results) / n

    return {
        "num_tasks": n,
        "k": k,
        "pass@1": pass_1,
        f"pass@{k}": pass_k,
        f"avg@{k}": avg_k,
        "accuracy": pass_1,
        "details": results,
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def evaluate_task_metrics(
    model,
    tokenizer,
    data_source: str | list[dict],
    task_type: str = "auto",
    max_samples: int | None = None,
    num_fewshot: int = 0,
    device: str = "cuda",
    batch_size: int = 4,
    # MCQ specific
    top_ns: list[int] | None = None,
    # Generation parameters
    k: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    repetition_penalty: float = 1.5,
) -> dict[str, Any]:
    """
    Unified evaluation entry point with batch inference.

    Args:
        model: Language model
        tokenizer: Tokenizer
        data_source: Path to JSONL or list of items
        task_type: "auto", "mcq", "generation", "humaneval"
        max_samples: Limit number of samples
        num_fewshot: Number of few-shot examples
        device: Device for inference
        batch_size: Batch size for inference
        top_ns: Top-n for MCQ
        k: Number of generations per item
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling p
        top_k: Top-k sampling
        repetition_penalty: Repetition penalty

    Returns:
        Dict with metrics and details
    """
    # Load data
    limit = max_samples + num_fewshot if max_samples else None
    items = (
        load_jsonl(data_source, limit=limit)
        if isinstance(data_source, str)
        else data_source
    )

    if not items:
        return {"error": "No items to evaluate"}

    # Handle few-shot split
    fewshot_items = []
    if num_fewshot > 0:
        fewshot_items = items[:num_fewshot]
        items = items[num_fewshot:]

    if max_samples:
        items = items[:max_samples]

    if not items:
        return {"error": "No items left to evaluate after few-shot split"}

    # Auto-detect task type
    if task_type == "auto":
        task_type = _detect_task_type(items[0])

    # Build generation config
    gen_config = GenerationConfig(
        k=k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    # Dispatch to appropriate evaluator
    if task_type == "humaneval":
        return evaluate_humaneval(
            model, tokenizer, items, gen_config, device, batch_size
        )
    elif task_type == "mcq":
        return evaluate_mcq(
            model, tokenizer, items, device, batch_size, top_ns, fewshot_items
        )
    elif task_type == "generation":
        return evaluate_generation(
            model, tokenizer, items, gen_config, device, batch_size, fewshot_items
        )
    else:
        return {"error": f"Unknown task type: {task_type}"}


def _detect_task_type(item: dict) -> str:
    """Auto-detect task type from item structure."""
    if all(k in item for k in ("prompt", "test", "entry_point")):
        return "humaneval"
    if "choices" in item and "label" in item:
        return "mcq"
    if any(k in item for k in ("solution", "answer", "answers")):
        return "generation"
    if "messages" in item:
        for m in item.get("messages", []):
            if m.get("role") == "assistant" and "####" in m.get("content", ""):
                return "generation"
    return "unknown"
