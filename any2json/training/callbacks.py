import difflib
import os
import random
from typing import Any, Callable
import json
import click
from datasets import load_from_disk, load_dataset, DatasetDict, Dataset
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, TrainingArguments
import wandb
from transformers.trainer_callback import TrainerCallback
from any2json.training.utils import (
    CausalLMDataCollator,
    process_raw_to_tokenized,
)
from any2json.utils import logger
from any2json.training.augment import Augmentor


class EvalLoggerCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        collator: CausalLMDataCollator,
        tokenized_eval_ds: Dataset,
        num_examples: int = 3,
        pad_to_multiple_of: int = 8,
        max_new_tokens: int = 8000,
    ):
        self.tokenizer = tokenizer
        self.collator = collator
        self.tokenized_eval_ds = tokenized_eval_ds
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_new_tokens = max_new_tokens
        self.table = wandb.Table(
            columns=[
                "epoch",
                "step",
                "prompt",
                "completion",
                "correct_completion",
                "diff",
                "sample_sequence",
            ],
            log_mode="INCREMENTAL",
        )
        self.num_examples = min(num_examples, len(tokenized_eval_ds))

    def sample_rows(self) -> list[dict[str, Any]]:
        idx = random.sample(range(len(self.tokenized_eval_ds)), self.num_examples)
        return [self.tokenized_eval_ds[i] for i in idx]

    def generate_completion_for_prompt(
        self,
        model: Any,
        generation_input: dict[str, list[int] | torch.Tensor],
    ) -> str:
        out = model.generate(
            **generation_input,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        seq = out[0]
        seq = seq[len(generation_input["input_ids"][0]) :]
        return self.tokenizer.decode(seq, skip_special_tokens=False)

    def generate_predictions(
        self, model: Any, tokenized: dict[str, torch.Tensor]
    ) -> tuple[list[str], list[str]]:
        """
        tokenized is the output of collator.
        It has 3 tensors of shape:
        - input_ids: [batch_size, seq_len]
        - labels: [batch_size, seq_len]
        - attention_mask: [batch_size, seq_len]
        """
        model.eval()
        device = next(model.parameters()).device
        inputs: list[str] = []
        preds: list[str] = []

        with torch.inference_mode():
            for input_ids, labels, attn in tqdm(
                zip(
                    tokenized["input_ids"],
                    tokenized["labels"],
                    tokenized["attention_mask"],
                    strict=True,
                ),
                desc="Generating eval example completions",
            ):
                start_idx = next(i for i, l in enumerate(labels) if l != -100)
                prompt_ids = input_ids[:start_idx]
                attn = attn[:start_idx]
                generation_input = {
                    "input_ids": prompt_ids.unsqueeze(0).to(device),
                    "attention_mask": attn.unsqueeze(0).to(device),
                }
                input_string = self.tokenizer.decode(
                    prompt_ids, skip_special_tokens=False
                )
                pred = self.generate_completion_for_prompt(
                    model,
                    generation_input,
                )
                inputs.append(input_string)
                preds.append(pred)
        return inputs, preds

    def log_examples(
        self,
        state: Any,
        tokenized_rows: list[dict[str, Any]],
        input_prompts: list[str],
        preds: list[str],
    ) -> None:
        for r, prompt, completion in zip(
            tokenized_rows, input_prompts, preds, strict=True
        ):
            input_data = self.tokenizer.decode(
                r["input_ids"], skip_special_tokens=False
            )
            correct_completion = input_data.split("[OUTPUT]")[1].strip()

            diff = difflib.ndiff(correct_completion, completion)

            self.table.add_data(
                state.epoch,
                state.global_step,
                prompt,
                completion,
                correct_completion,
                diff,
                input_data,
            )
        wandb.log({"eval_examples": self.table})

    def on_evaluate(
        self, args, state, control, model=None, tokenizer=None, metrics=None, **kwargs
    ):
        logger.info(f"Running eval example predictions callback")
        if model is None:
            return
        tokenized_rows = self.sample_rows()
        collated = self.collator(tokenized_rows)
        input_prompts, preds = self.generate_predictions(model, collated)
        logger.info(
            f"Generated {len(input_prompts)} predictions.\nPrompts:\n{input_prompts}\nPreds:\n{preds}"
        )
        self.log_examples(state, tokenized_rows, input_prompts, preds)


class DebugTokensCallback(TrainerCallback):
    def __init__(self, tokenizer: AutoTokenizer, max_examples: int = 1):
        self.tokenizer = tokenizer
        self.max_examples = max_examples
        self.step_count = 0

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_count += 1

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if hasattr(model, "forward"):
            original_forward = model.forward

            def debug_forward(
                input_ids=None, attention_mask=None, labels=None, **kwargs
            ):
                if self.step_count <= 3:
                    logger.info(f"DEBUG FORWARD PASS - Step {self.step_count}:")
                    if input_ids is not None:
                        logger.info(f"Input IDs shape: {input_ids.shape}")
                        lengths = [
                            len(
                                [
                                    t
                                    for t in example_ids
                                    if t != self.tokenizer.pad_token_id
                                ]
                            )
                            for example_ids in input_ids.tolist()
                        ]
                        logger.info(f"Input lengths in batch: {lengths}")
                        for i in range(min(self.max_examples, input_ids.shape[0])):
                            example_ids = input_ids[i].tolist()
                            example_mask = (
                                attention_mask[i].tolist()
                                if attention_mask is not None
                                else None
                            )
                            example_labels = (
                                labels[i].tolist() if labels is not None else None
                            )

                            logger.info(f"  Example {i}:")
                            logger.info(f"    Input IDs: {example_ids}")
                            logger.info(
                                f"    Input decoded: {repr(self.tokenizer.decode(example_ids))}"
                            )
                            example_ids_no_pad = [
                                t
                                for t in example_ids
                                if t != self.tokenizer.pad_token_id
                            ]
                            logger.info(f"    Input length: {len(example_ids_no_pad)}")
                            if example_mask:
                                logger.info(f"    Attention mask: {example_mask}")
                            if example_labels:
                                target_tokens = [t for t in example_labels if t != -100]
                                logger.info(f"    Label tokens: {example_labels}")
                                logger.info(
                                    f"    Target tokens ({len(target_tokens)}): {target_tokens}"
                                )
                                if target_tokens:
                                    logger.info(
                                        f"    Target decoded: {repr(self.tokenizer.decode(target_tokens))}"
                                    )

                return original_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    **kwargs,
                )

            model.forward = debug_forward
