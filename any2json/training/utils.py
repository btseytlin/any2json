import os
import random
from typing import Any, Callable

import click
from datasets import load_from_disk, load_dataset, DatasetDict
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import wandb
from transformers.trainer_callback import TrainerCallback
from any2json.training.augment import apply_augmentations, build_augmentor
from any2json.utils import logger
from any2json.grouping import train_test_split_groups


def resolve_pad_id(tokenizer: AutoTokenizer) -> int:
    return tokenizer.pad_token_id or tokenizer.eos_token_id or tokenizer.unk_token_id


def format_example(input_data: str, schema: str, output: str = "") -> str:
    system_prompt = "Convert input data to json according to JSONSchema"
    return f"{system_prompt}\n[SCHEMA]{schema}[INPUT]{input_data}[OUTPUT]{output}"


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    v = sorted(values)
    k = int((len(v) - 1) * q)
    return v[k]


def make_group_split(ds_dict: DatasetDict, test_size: int, seed: int) -> DatasetDict:
    groups = [s["meta"]["group"] for s in ds_dict["train"]]
    items = list(range(len(groups)))
    train_idx, test_idx, _, _ = train_test_split_groups(
        items, groups, test_size=test_size, random_state=seed
    )
    train_split = ds_dict["train"].select(train_idx)
    eval_split = ds_dict["train"].select(test_idx)
    return DatasetDict({"train": train_split, "validation": eval_split})


def load_hf_dataset(path_or_repo: str) -> DatasetDict:
    if os.path.isdir(path_or_repo):
        return load_from_disk(path_or_repo)
    return load_dataset(path_or_repo)


def apply_debug_limit(ds: DatasetDict, limit: int) -> DatasetDict:
    if limit <= 0 or "train" not in ds:
        return ds
    n = min(limit, len(ds["train"]))
    return DatasetDict({"train": ds["train"].select(range(n))})


def ids_to_token_str(tok: AutoTokenizer, ids: list[int]) -> str:
    out: list[str] = []
    for t in ids:
        if t == -100:
            out.append("-100")
        else:
            out.append(tok.convert_ids_to_tokens([t], skip_special_tokens=False)[0])
    return " ".join(out)


def estimate_token_lengths(dataset_path: str, model_name: str, samples: int) -> None:
    ds_all = load_hf_dataset(dataset_path)
    base = ds_all["train"] if "train" in ds_all else list(ds_all.values())[0]
    n = min(samples, len(base))
    rows = [base[i] for i in range(n)]
    tok = AutoTokenizer.from_pretrained(model_name)
    src = [
        len(tok(format_example(r["input_data"], r["schema"]))["input_ids"])
        for r in rows
    ]
    tgt = [len(tok(r["output"])["input_ids"]) for r in rows]
    click.echo(
        f"src p90={percentile(src, 0.9)} p95={percentile(src, 0.95)} p99={percentile(src, 0.99)}"
    )
    click.echo(
        f"tgt p90={percentile(tgt, 0.9)} p95={percentile(tgt, 0.95)} p99={percentile(tgt, 0.99)}"
    )


def build_train_sequence(
    tokenizer: AutoTokenizer,
    prompt_ids: list[int],
    target_ids: list[int],
) -> tuple[list[int], list[int]]:
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must define eos_token_id")
    ids = [bos] + prompt_ids + target_ids + [eos]
    labels = ([-100] * (len(prompt_ids) + 1)) + target_ids + [eos]
    return ids, labels


def build_tokenize_fn(
    tokenizer: AutoTokenizer,
    debug: bool = False,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must define eos_token_id")

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        input_data = batch["input_data"]
        schema = batch["schema"]
        output = batch["output"]
        prompts = [
            format_example(i, s) for i, s in zip(input_data, schema, strict=True)
        ]
        enc_in = tokenizer(prompts, add_special_tokens=False)
        enc_out = tokenizer(output, add_special_tokens=False)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        lengths: list[int] = []
        for idx, (prompt_ids, target_ids) in enumerate(
            zip(enc_in["input_ids"], enc_out["input_ids"], strict=True)
        ):
            ids, lbs = build_train_sequence(tokenizer, prompt_ids, target_ids)
            input_ids.append(ids)
            labels.append(lbs)
            lengths.append(len(ids))
            if debug and idx == 0:
                logger.debug(f"DEBUG TOKENIZATION - Example {idx}:")
                logger.debug(f"  Input prompt: {repr(prompts[idx])}")
                logger.debug(f"  Input tokens: {prompt_ids}")
                logger.debug(f"  Input decoded: {repr(tokenizer.decode(prompt_ids))}")
                logger.debug(f"  Target tokens: {target_ids}")
                logger.debug(f"  Target decoded: {repr(tokenizer.decode(target_ids))}")
                logger.debug(f"  Output: {repr(batch['output'][idx])}")
                logger.debug(f"  EOS token: {eos}")
                logger.debug(f"  Final input_ids: {ids}")
                logger.debug(f"  Final labels: {lbs}")
                logger.debug(f"  Final length: {len(ids)}")
        return {"input_ids": input_ids, "labels": labels, "length": lengths}

    return tokenize


def pad_to_multiple(
    ids: list[list[int]], multiple: int, pad_id: int
) -> tuple[list[list[int]], list[list[int]]]:
    max_len = max(len(i) for i in ids)
    max_len = ((max_len + multiple - 1) // multiple) * multiple
    out = []
    attn = []
    for i in ids:
        pad = max(0, max_len - len(i))
        out.append(i + [pad_id] * pad)
        attn.append([1] * len(i) + [0] * pad)
    return out, attn


class CausalLMDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_id = resolve_pad_id(self.tokenizer)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        input_ids, attention_mask = pad_to_multiple(
            [f["input_ids"] for f in features],
            self.pad_to_multiple_of,
            self.pad_id,
        )
        labels, _ = pad_to_multiple(
            [f["labels"] for f in features],
            self.pad_to_multiple_of,
            -100,
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def encode_prompt_ids(
    tokenizer: AutoTokenizer, input_data: str, schema: str
) -> list[int]:
    enc = tokenizer(format_example(input_data, schema), add_special_tokens=False)
    return enc["input_ids"]


def encode_target_ids(tokenizer: AutoTokenizer, output: str) -> list[int]:
    enc = tokenizer(output, add_special_tokens=False)
    return enc["input_ids"]


# TODO WTF is that
# def build_gen_toks(
#     device: torch.device,
#     prompt_ids: list[int],
#     pad_multiple: int,
#     pad_id: int,
# ) -> tuple[dict[str, torch.Tensor], list[int], list[int]]:
#     padded, attn = pad_to_multiple(prompt_ids, pad_multiple, pad_id)
#     toks = {
#         "input_ids": torch.tensor([padded], dtype=torch.long).to(device),
#         "attention_mask": torch.tensor([attn], dtype=torch.long).to(device),
#     }
#     return toks, padded, attn


def build_tokenized_length_filter_fn(
    max_sequence_length: int,
) -> Callable[[dict[str, Any]], list[bool]]:
    def pred(batch: dict[str, Any]) -> list[bool]:
        return [len(seq) <= max_sequence_length for seq in batch["input_ids"]]

    return pred


def filter_tokenized_splits_by_length(
    ds: DatasetDict,
    max_sequence_length: int,
    num_proc: int = 8,
) -> DatasetDict:
    pred = build_tokenized_length_filter_fn(max_sequence_length)
    train_f = ds["train"].filter(pred, batched=True, num_proc=num_proc)
    val_f = ds["validation"].filter(pred, batched=True, num_proc=num_proc)
    return DatasetDict({"train": train_f, "validation": val_f})


def augment_train_split(ds: DatasetDict, cfg, seed: int) -> DatasetDict:
    aug = build_augmentor(
        drop_schema_proba=cfg.drop_schema_proba,
        schema_missing_token=cfg.schema_missing_token,
        input_aug_paths=cfg.input_aug,
        output_aug_paths=cfg.output_aug,
    )
    rng = random.Random(seed)

    def map_fn(batch: dict[str, Any]) -> dict[str, Any]:
        inputs, schemas, outputs = [], [], []
        for i, s, o in zip(
            batch["input_data"], batch["schema"], batch["output"], strict=True
        ):
            ni, ns, no = apply_augmentations(i, s, o, aug, rng.random)
            inputs.append(ni)
            schemas.append(ns)
            outputs.append(no)
        return {
            "input_data": inputs,
            "schema": schemas,
            "output": outputs,
            "meta": batch.get("meta"),
        }

    train_aug = ds["train"].map(map_fn, batched=True)
    return DatasetDict({"train": train_aug, "validation": ds["validation"]})


class EvalLoggerCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        raw_eval_ds,
        num_examples: int = 3,
        pad_to_multiple_of: int = 8,
        max_new_tokens: int = 50,
    ):
        self.tokenizer = tokenizer
        self.raw_eval_ds = raw_eval_ds
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_new_tokens = max_new_tokens
        self.table = wandb.Table(
            columns=[
                "epoch",
                "step",
                "prompt",
                "completion",
                "input",
                "schema",
                "target",
            ],
            log_mode="INCREMENTAL",
        )
        self.num_examples = min(num_examples, len(raw_eval_ds))
        self.tokenize_fn = build_tokenize_fn(self.tokenizer)

    def sample_rows(self) -> list[dict[str, Any]]:
        idx = random.sample(range(len(self.raw_eval_ds)), self.num_examples)
        return [self.raw_eval_ds[i] for i in idx]

    def generate_completion_for_prompt(
        self,
        model: Any,
        generation_input: dict[str, Any],
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
        self, model: Any, tokenized: dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        model.eval()
        device = next(model.parameters()).device
        inputs: list[str] = []
        preds: list[str] = []
        with torch.inference_mode():
            for input_ids, labels in tqdm(
                zip(tokenized["input_ids"], tokenized["labels"], strict=True),
                desc="Generating eval example completions",
            ):
                start_idx = next(i for i, l in enumerate(labels) if l != -100)
                prompt_ids = input_ids[:start_idx]
                padded, attn = pad_to_multiple(
                    [prompt_ids],
                    self.pad_to_multiple_of,
                    resolve_pad_id(self.tokenizer),
                )
                generation_input = {
                    "input_ids": torch.tensor(padded, dtype=torch.long).to(device),
                    "attention_mask": torch.tensor(attn, dtype=torch.long).to(device),
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
        rows: list[dict[str, Any]],
        input_prompts: list[str],
        preds: list[str],
    ) -> None:
        for r, prompt, completion in zip(rows, input_prompts, preds, strict=True):
            self.table.add_data(
                state.epoch,
                state.global_step,
                prompt,
                completion,
                r["input_data"],
                r["schema"],
                r["output"],
            )
        wandb.log({"eval_examples": self.table})

    def on_evaluate(
        self, args, state, control, model=None, tokenizer=None, metrics=None, **kwargs
    ):
        logger.info(f"Running eval example predictions callback")
        if model is None:
            return
        rows = self.sample_rows()
        tokenized = self.tokenize_fn(rows)
        input_prompts, preds = self.generate_predictions(model, tokenized)
        logger.info(
            f"Generated {len(input_prompts)} predictions.\nPrompts:\n{input_prompts}\nPreds:\n{preds}"
        )
        self.log_examples(state, rows, input_prompts, preds)


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
