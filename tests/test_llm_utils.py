import json
import os
from datasets import Dataset
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

from any2json.training.train import PipelineConfig, prepare_model_and_tokenizer
from any2json.training.utils import (
    format_example,
    pad_to_multiple,
    build_train_sequence,
    build_tokenize_fn,
    CausalLMDataCollator,
    process_raw_to_tokenized,
    resolve_pad_id,
    build_tokenized_length_filter_fn,
    ids_to_token_str,
)
from any2json.training.callbacks import EvalLoggerCallback

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture(scope="session")
def model_and_tokenizer():
    pcfg = PipelineConfig()
    args = TrainingArguments()
    pcfg.hf_args = args
    pcfg.model_name = "HuggingFaceTB/SmolLM2-135M"
    model, tokenizer = prepare_model_and_tokenizer(
        pcfg,
        args,
    )
    return model, tokenizer


@pytest.fixture
def tokenizer(model_and_tokenizer):
    return model_and_tokenizer[1]


@pytest.fixture
def model(model_and_tokenizer):
    return model_and_tokenizer[0]


def test_format_example() -> None:
    s = format_example("input", "schema")
    expected = "Convert input data to json according to JSONSchema\n[SCHEMA]schema[INPUT]input[OUTPUT]"
    assert s == expected

    s = format_example("input", "schema", "output")
    expected = "Convert input data to json according to JSONSchema\n[SCHEMA]schema[INPUT]input[OUTPUT]output"
    assert s == expected


def test_resolve_pad_id(tokenizer) -> None:
    class A:
        pad_token_id = 7
        eos_token_id = 1
        unk_token_id = 2

    class B:
        pad_token_id = None
        eos_token_id = 5
        unk_token_id = 9

    class C:
        pad_token_id = None
        eos_token_id = None
        unk_token_id = 3

    assert resolve_pad_id(A()) == 7
    assert resolve_pad_id(B()) == 5
    assert resolve_pad_id(C()) == 3
    pad_id = resolve_pad_id(tokenizer)
    assert pad_id is not None


class TestPadToMultiple:
    def test_pad_to_multiple_no_pad(self) -> None:
        ids, attn = pad_to_multiple(
            ids=[[9, 8, 7]],
            multiple=3,
            pad_id=0,
        )
        assert ids == [[9, 8, 7]]
        assert attn == [[1, 1, 1]]

    def test_pad_to_multiple_base(self) -> None:
        ids, attn = pad_to_multiple(
            ids=[[9, 8, 7]],
            multiple=4,
            pad_id=0,
        )
        assert ids == [[9, 8, 7, 0]]
        assert attn == [[1, 1, 1, 0]]

    def test_pad_to_multiple_larger_input(self) -> None:
        ids, attn = pad_to_multiple(
            ids=[[9, 8, 7, 6, 5, 4]],
            multiple=4,
            pad_id=0,
        )
        assert ids == [[9, 8, 7, 6, 5, 4, 0, 0]]
        assert attn == [[1, 1, 1, 1, 1, 1, 0, 0]]

    def test_pad_to_multiple_eight(self) -> None:
        ids, attn = pad_to_multiple(
            ids=[[9, 8, 7]],
            multiple=8,
            pad_id=0,
        )
        assert ids == [[9, 8, 7, 0, 0, 0, 0, 0]]
        assert attn == [[1, 1, 1, 0, 0, 0, 0, 0]]

    def test_pad_to_multiple_multiple_inputs(self) -> None:
        ids, attn = pad_to_multiple(
            ids=[[9, 8, 7, 6], [6, 5, 4]],
            multiple=4,
            pad_id=0,
        )
        assert ids == [[9, 8, 7, 6], [6, 5, 4, 0]]
        assert attn == [[1, 1, 1, 1], [1, 1, 1, 0]]


class TestTokenization:
    def test_build_train_sequence_base(self, tokenizer) -> None:
        prompt_ids = [3, 4]
        target_ids = [5]
        ids, labels = build_train_sequence(tokenizer, prompt_ids, target_ids)
        assert len(ids) == len(labels)
        assert ids == [
            tokenizer.bos_token_id,
            *prompt_ids,
            *target_ids,
            tokenizer.eos_token_id,
        ]
        assert labels == [
            -100,
            *([-100] * len(prompt_ids)),
            *target_ids,
            tokenizer.eos_token_id,
        ]

    def test_build_tokenize_fn_one_example_in_batch(self, tokenizer) -> None:
        fn = build_tokenize_fn(tokenizer)
        batch = {
            "input_data": ["hello world"],
            "schema": [
                '{"type": "object", "properties": {"text": {"type": "string"}}}'
            ],
            "output": ['{"text": "hello world"}'],
        }
        out = fn(batch)
        assert len(out["input_ids"]) == 1
        assert len(out["labels"]) == 1
        assert len(out["length"]) == 1

        assert len(out["input_ids"][0]) == out["length"][0]
        assert out["input_ids"][0][0] == tokenizer.bos_token_id
        assert out["input_ids"][0][-1] == tokenizer.eos_token_id
        assert out["labels"][0][0] == -100
        assert out["labels"][0][-1] == tokenizer.eos_token_id

        input_ids = [l for l in out["input_ids"][0]]
        assert (
            tokenizer.decode(input_ids)
            == f"{tokenizer.bos_token}Convert input data to json according to JSONSchema\n[SCHEMA]{batch['schema'][0]}[INPUT]hello world[OUTPUT]{batch['output'][0]}{tokenizer.eos_token}"
        )

        target_ids = [l for l in out["labels"][0] if l != -100]
        expected_target = f"{batch['output'][0]}{tokenizer.eos_token}"
        assert tokenizer.decode(target_ids) == expected_target

    def test_build_tokenize_fn_two_examples_in_batch(self, tokenizer) -> None:
        fn = build_tokenize_fn(tokenizer)
        batch = {
            "input_data": [
                "hello world",
                "goodbye world",
            ],
            "schema": [
                '{"type": "object", "properties": {"text": {"type": "string"}}}',
                '{"type": "object", "properties": {"text": {"type": "string"}}}',
            ],
            "output": [
                '{"text": "hello world"}',
                '{"text": "goodbye world"}',
            ],
        }
        out = fn(batch)
        assert len(out["input_ids"]) == 2
        assert len(out["labels"]) == 2
        assert len(out["length"]) == 2

        for i in range(2):
            assert len(out["input_ids"][i]) == out["length"][i]
            assert out["input_ids"][i][0] == tokenizer.bos_token_id
            assert out["input_ids"][i][-1] == tokenizer.eos_token_id
            assert out["labels"][i][0] == -100
            assert out["labels"][i][-1] == tokenizer.eos_token_id

        for i in range(2):
            input_ids = [l for l in out["input_ids"][i]]
            assert (
                tokenizer.decode(input_ids)
                == f"{tokenizer.bos_token}Convert input data to json according to JSONSchema\n[SCHEMA]{batch['schema'][i]}[INPUT]{batch['input_data'][i]}[OUTPUT]{batch['output'][i]}{tokenizer.eos_token}"
            )

            target_ids = [l for l in out["labels"][i] if l != -100]
            expected_target = f"{batch['output'][i]}{tokenizer.eos_token}"
            assert tokenizer.decode(target_ids) == expected_target

    def test_causal_lm_data_collator_padding(self, tokenizer) -> None:
        coll = CausalLMDataCollator(tokenizer, pad_to_multiple_of=4)
        features = [
            {"input_ids": [3, 4, 5, 0], "labels": [-100, -100, 7, 0], "length": 3},
            {
                "input_ids": [6, 7, 8, 9, 10],
                "labels": [-100, -100, 1, 2, 3],
                "length": 5,
            },
        ]
        out = coll(features)
        assert out["input_ids"].shape == torch.Size([2, 8])

        assert out["input_ids"][0].tolist() == [
            3,
            4,
            5,
            0,
            0,
            0,
            0,
            0,
        ]
        assert out["attention_mask"][0].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]

        assert out["input_ids"][1].tolist() == [
            6,
            7,
            8,
            9,
            10,
            0,
            0,
            0,
        ]
        assert out["attention_mask"][1].tolist() == [1, 1, 1, 1, 1, 0, 0, 0]

    def test_build_tokenized_length_filter_fn(self) -> None:
        pred = build_tokenized_length_filter_fn(
            max_sequence_length=3,
        )
        batch = {
            "input_ids": [[1, 2, 3], [1, 2, 3, 4, 5]],
            "labels": [[-100, -100, 7], [-100, 7, 8, 9, -100]],
        }
        res = pred(batch)
        assert res == [True, False]


def test_ids_to_token_str_and_encoders(tokenizer) -> None:
    hello_ids = tokenizer("hello", add_special_tokens=False)["input_ids"]
    if hello_ids:
        wid = hello_ids[0]
        s = ids_to_token_str(tokenizer, [-100, wid, -100])
        assert "-100" in s


def test_full_pipeline_with_real_tokenizer(tokenizer) -> None:
    fn = build_tokenize_fn(tokenizer)
    coll = CausalLMDataCollator(tokenizer, pad_to_multiple_of=8)

    batch = {
        "input_data": ["translate to json", "convert data"],
        "schema": ["object", "array"],
        "output": ['{"text": "hello"}', '["item1", "item2"]'],
    }

    tokenized = fn(batch)
    collated = coll(
        [
            {"input_ids": tokenized["input_ids"][0], "labels": tokenized["labels"][0]},
            {"input_ids": tokenized["input_ids"][1], "labels": tokenized["labels"][1]},
        ]
    )

    assert collated["input_ids"].shape[0] == 2
    assert collated["labels"].shape == collated["input_ids"].shape
    assert collated["attention_mask"].shape == collated["input_ids"].shape

    for i in range(2):
        seq_len = len(tokenized["input_ids"][i])
        assert torch.all(collated["attention_mask"][i, :seq_len] == 1)
        if collated["input_ids"].shape[1] > seq_len:
            assert torch.all(collated["attention_mask"][i, seq_len:] == 0)


@pytest.fixture
def mock_eval_dataset():
    return Dataset.from_list(
        [
            {
                "input_data": "hello world",
                "schema": '{"type": "object", "properties": {"text": {"type": "string"}}}',
                "output": '{"text": "hello world"}',
            },
            {
                "input_data": "test data",
                "schema": '{"type": "object", "properties": {"message": {"type": "string"}}}',
                "output": '{"message": "test data"}',
            },
        ]
    )


@pytest.fixture
def tokenized_mock_eval_dataset(tokenizer, mock_eval_dataset):
    tokenize_fn = build_tokenize_fn(tokenizer)
    tokenized_eval_ds = process_raw_to_tokenized(
        dataset=mock_eval_dataset,
        tokenize_fn=tokenize_fn,
        filter_fn=lambda batch: [True for _ in batch],
    )

    return tokenized_eval_ds


class TestEvalLoggerCallback:
    def test_init(self, tokenizer, tokenized_mock_eval_dataset):
        callback = EvalLoggerCallback(
            tokenizer=tokenizer,
            collator=CausalLMDataCollator(tokenizer, pad_to_multiple_of=8),
            tokenized_eval_ds=tokenized_mock_eval_dataset,
            num_examples=2,
            pad_to_multiple_of=8,
            max_new_tokens=10,
        )
        assert callback.tokenizer == tokenizer
        assert callback.pad_to_multiple_of == 8
        assert callback.num_examples == 2

    def test_sample_rows(self, tokenizer, tokenized_mock_eval_dataset):
        callback = EvalLoggerCallback(
            tokenizer=tokenizer,
            collator=CausalLMDataCollator(tokenizer, pad_to_multiple_of=8),
            tokenized_eval_ds=tokenized_mock_eval_dataset,
            num_examples=2,
        )
        rows = callback.sample_rows()
        assert len(rows) == 2
        assert all(isinstance(row, dict) for row in rows)
        assert all("input_ids" in row for row in rows), rows

    def test_generate_completion_for_prompt(
        self, tokenizer, model, tokenized_mock_eval_dataset
    ):
        callback = EvalLoggerCallback(
            tokenizer=tokenizer,
            collator=CausalLMDataCollator(tokenizer, pad_to_multiple_of=8),
            tokenized_eval_ds=tokenized_mock_eval_dataset,
            num_examples=1,
            max_new_tokens=10,
        )

        prompt_ids = tokenizer("test prompt", add_special_tokens=False)["input_ids"]
        padded, attn = pad_to_multiple([prompt_ids], 8, resolve_pad_id(tokenizer))
        toks = {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

        pred = callback.generate_completion_for_prompt(model, toks)
        assert pred == "\n\nThe prompt is a text box that you"

    def test_generate_predictions(self, tokenizer, model, tokenized_mock_eval_dataset):
        callback = EvalLoggerCallback(
            tokenizer=tokenizer,
            collator=CausalLMDataCollator(tokenizer, pad_to_multiple_of=8),
            tokenized_eval_ds=tokenized_mock_eval_dataset,
            num_examples=1,
            max_new_tokens=10,
        )

        features = [
            {
                "input_ids": tokenized_mock_eval_dataset[0]["input_ids"],
                "labels": tokenized_mock_eval_dataset[0]["labels"],
            },
            {
                "input_ids": tokenized_mock_eval_dataset[1]["input_ids"],
                "labels": tokenized_mock_eval_dataset[1]["labels"],
            },
        ]

        batch = callback.collator(features)
        inputs, preds = callback.generate_predictions(model, batch)

        assert (
            inputs[0]
            == '<|endoftext|>Convert input data to json according to JSONSchema\n[SCHEMA]{"type": "object", "properties": {"text": {"type": "string"}}}[INPUT]hello world[OUTPUT]'
        )
        assert preds[0].strip() == "The JSONSchema is a specification for the"

        assert (
            inputs[1]
            == '<|endoftext|>Convert input data to json according to JSONSchema\n[SCHEMA]{"type": "object", "properties": {"message": {"type": "string"}}}[INPUT]test data[OUTPUT]'
        )
        # assert preds[1].strip() == "json data The JSONSchema is a standard"

        assert len(preds) == 2
        assert "SCHEMA" not in preds[0]
        assert "INPUT" not in preds[0]
