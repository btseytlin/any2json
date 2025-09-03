import json
import os
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


from transformers import AutoTokenizer
from any2json.training.utils import format_chat_messages, build_tokenize_fn
from any2json.training.utils import (
    format_example,
    pad_to_multiple,
    build_train_sequence,
    build_tokenize_fn,
    CausalLMDataCollator,
    resolve_pad_id,
    build_tokenized_length_filter_fn,
    ids_to_token_str,
    encode_prompt_ids,
    encode_target_ids,
    load_tokenizer,
)
from any2json.training.callbacks import EvalLoggerCallback


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture
def tokenizer():
    tokenizer = load_tokenizer("google/gemma-3-270m", use_chat_template=True)
    return tokenizer


@pytest.fixture
def model():
    return AutoModelForCausalLM.from_pretrained("google/gemma-3-270m")


def test_format_example() -> None:
    s = format_example("input", "schema")
    expected = "Convert input data to json according to JSONSchema\n[SCHEMA]schema[INPUT]input[OUTPUT]"
    assert s == expected

    s = format_example("input", "schema", "output")
    expected = "Convert input data to json according to JSONSchema\n[SCHEMA]schema[INPUT]input[OUTPUT]output"
    assert s == expected


def test_format_chat_messages() -> None:
    messages = format_chat_messages("input", "schema")
    expected = [
        {
            "role": "system",
            "content": "Convert input data to json according to JSONSchema",
        },
        {"role": "user", "content": "[SCHEMA]schema[INPUT]input"},
    ]
    assert messages == expected

    messages = format_chat_messages("input", "schema", "output")
    expected = [
        {
            "role": "system",
            "content": "Convert input data to json according to JSONSchema",
        },
        {"role": "user", "content": "[SCHEMA]schema[INPUT]input"},
        {"role": "assistant", "content": "output"},
    ]
    assert messages == expected

    input_dict = {"name": "John", "age": 30}
    schema_dict = {"type": "object", "properties": {"name": {"type": "string"}}}
    output_dict = {"name": "John", "age": 30}

    messages = format_chat_messages(input_dict, schema_dict, output_dict)
    assert (
        messages[1]["content"]
        == '[SCHEMA]{"type":"object","properties":{"name":{"type":"string"}}}[INPUT]{"name":"John","age":30}'
    )
    assert messages[2]["content"] == '{"name":"John","age":30}'

    messages = format_chat_messages("input", None, "output")
    assert "[MISSING]" == messages[1]["content"]


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
        decoded = tokenizer.decode(input_ids)
        expected_input = f"{tokenizer.bos_token}<start_of_turn>model\nConvert input data to json according to JSONSchema<end_of_turn>\n<start_of_turn>user\n[SCHEMA]{batch['schema'][0]}[INPUT]hello world<end_of_turn>\n<start_of_turn>model\n{batch['output'][0]}<end_of_turn>\n{tokenizer.eos_token}"
        assert decoded == expected_input

        target_ids = [l for l in out["labels"][0] if l != -100]
        target_decoded = tokenizer.decode(target_ids)
        expected_target = f"{batch['output'][0]}<end_of_turn>\n{tokenizer.eos_token}"
        assert target_decoded == expected_target

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
            decoded = tokenizer.decode(input_ids)
            expected_input = f"{tokenizer.bos_token}<start_of_turn>model\nConvert input data to json according to JSONSchema<end_of_turn>\n<start_of_turn>user\n[SCHEMA]{batch['schema'][i]}[INPUT]{batch['input_data'][i]}<end_of_turn>\n<start_of_turn>model\n{batch['output'][i]}<end_of_turn>\n{tokenizer.eos_token}"
            assert decoded == expected_input

            target_ids = [l for l in out["labels"][i] if l != -100]
            target_decoded = tokenizer.decode(target_ids)
            expected_target = (
                f"{batch['output'][i]}<end_of_turn>\n{tokenizer.eos_token}"
            )
            assert target_decoded == expected_target

    def test_causal_lm_data_collator_padding(self, tokenizer) -> None:
        pad_id = resolve_pad_id(tokenizer)
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
            pad_id,
            pad_id,
            pad_id,
            pad_id,
        ]
        assert out["attention_mask"][0].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]

        assert out["input_ids"][1].tolist() == [6, 7, 8, 9, 10, pad_id, pad_id, pad_id]
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

    a = encode_prompt_ids(tokenizer, "in", "sch")
    expected_prompt = "<start_of_turn>model\nConvert input data to json according to JSONSchema<end_of_turn>\n<start_of_turn>user\n[SCHEMA]sch[INPUT]in<end_of_turn>\n<start_of_turn>model\n"
    expected_ids = tokenizer(expected_prompt, add_special_tokens=False)["input_ids"]
    assert a == expected_ids

    c = encode_target_ids(tokenizer, "out")
    d = tokenizer("out", add_special_tokens=False)["input_ids"]
    assert c == d


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
    return [
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


class TestEvalLoggerCallback:
    def test_init(self, tokenizer, mock_eval_dataset):
        tokenize_fn = build_tokenize_fn(tokenizer)
        batch = {
            "input_data": [d["input_data"] for d in mock_eval_dataset],
            "schema": [d["schema"] for d in mock_eval_dataset],
            "output": [d["output"] for d in mock_eval_dataset],
        }
        tokenized_eval_ds = tokenize_fn(batch)
        collator = CausalLMDataCollator(tokenizer, pad_to_multiple_of=8)
        callback = EvalLoggerCallback(
            tokenizer=tokenizer,
            collator=collator,
            tokenized_eval_ds=tokenized_eval_ds,
            num_examples=2,
            pad_to_multiple_of=8,
        )
        assert callback.tokenizer == tokenizer
        assert callback.tokenized_eval_ds == tokenized_eval_ds
        assert callback.pad_to_multiple_of == 8
        assert callback.num_examples == 2

    def test_sample_rows(self, tokenizer, mock_eval_dataset):
        tokenize_fn = build_tokenize_fn(tokenizer)
        batch = {
            "input_data": [d["input_data"] for d in mock_eval_dataset],
            "schema": [d["schema"] for d in mock_eval_dataset],
            "output": [d["output"] for d in mock_eval_dataset],
        }
        tokenized_result = tokenize_fn(batch)
        # Create a mock dataset from the tokenized result
        tokenized_eval_ds = [
            {
                "input_ids": tokenized_result["input_ids"][i],
                "labels": tokenized_result["labels"][i],
            }
            for i in range(len(tokenized_result["input_ids"]))
        ]
        collator = CausalLMDataCollator(tokenizer, pad_to_multiple_of=8)
        callback = EvalLoggerCallback(
            tokenizer=tokenizer,
            collator=collator,
            tokenized_eval_ds=tokenized_eval_ds,
            num_examples=2,
        )
        rows = callback.sample_rows()
        assert len(rows) == 2
        assert all(isinstance(row, dict) for row in rows)

    def test_generate_completion_for_prompt(self, tokenizer, model, mock_eval_dataset):
        tokenize_fn = build_tokenize_fn(tokenizer)
        batch = {
            "input_data": [d["input_data"] for d in mock_eval_dataset],
            "schema": [d["schema"] for d in mock_eval_dataset],
            "output": [d["output"] for d in mock_eval_dataset],
        }
        tokenized_result = tokenize_fn(batch)
        tokenized_eval_ds = [
            {
                "input_ids": tokenized_result["input_ids"][i],
                "labels": tokenized_result["labels"][i],
            }
            for i in range(len(tokenized_result["input_ids"]))
        ]
        collator = CausalLMDataCollator(tokenizer, pad_to_multiple_of=8)
        callback = EvalLoggerCallback(
            tokenizer=tokenizer,
            collator=collator,
            tokenized_eval_ds=tokenized_eval_ds,
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

    def test_generate_predictions(self, tokenizer, model, mock_eval_dataset):
        tokenize_fn = build_tokenize_fn(tokenizer)
        batch = {
            "input_data": [d["input_data"] for d in mock_eval_dataset],
            "schema": [d["schema"] for d in mock_eval_dataset],
            "output": [d["output"] for d in mock_eval_dataset],
        }
        tokenized_result = tokenize_fn(batch)
        tokenized_eval_ds = [
            {
                "input_ids": tokenized_result["input_ids"][i],
                "labels": tokenized_result["labels"][i],
            }
            for i in range(len(tokenized_result["input_ids"]))
        ]
        collator = CausalLMDataCollator(tokenizer, pad_to_multiple_of=8)
        callback = EvalLoggerCallback(
            tokenizer=tokenizer,
            collator=collator,
            tokenized_eval_ds=tokenized_eval_ds,
            num_examples=1,
            max_new_tokens=10,
        )

        # Use the tokenized dataset directly
        inputs, preds = callback.generate_predictions(
            model,
            {
                "input_ids": [tokenized_result["input_ids"][0]],
                "labels": [tokenized_result["labels"][0]],
            },
        )

        assert inputs == [
            '<|endoftext|>Convert input data to json according to JSONSchema\n[SCHEMA]{"type": "object", "properties": {"text": {"type": "string"}}}[INPUT]hello world[OUTPUT]'
        ]

        assert isinstance(preds[0], str)
        assert len(preds[0]) > 0
        assert "SCHEMA" not in preds[0]
        assert "INPUT" not in preds[0]
        assert batch["input_data"][0] not in preds[0]
        assert batch["schema"][0] not in preds[0]

    def test_generate_predictions_multiple_examples(
        self, tokenizer, model, mock_eval_dataset
    ):
        tokenize_fn = build_tokenize_fn(tokenizer)
        batch = {
            "input_data": [d["input_data"] for d in mock_eval_dataset],
            "schema": [d["schema"] for d in mock_eval_dataset],
            "output": [d["output"] for d in mock_eval_dataset],
        }
        tokenized_result = tokenize_fn(batch)
        tokenized_eval_ds = [
            {
                "input_ids": tokenized_result["input_ids"][i],
                "labels": tokenized_result["labels"][i],
            }
            for i in range(len(tokenized_result["input_ids"]))
        ]
        collator = CausalLMDataCollator(tokenizer, pad_to_multiple_of=8)
        callback = EvalLoggerCallback(
            tokenizer=tokenizer,
            collator=collator,
            tokenized_eval_ds=tokenized_eval_ds,
            num_examples=2,
            pad_to_multiple_of=8,
            max_new_tokens=10,
        )

        inputs, preds = callback.generate_predictions(
            model,
            {
                "input_ids": tokenized_result["input_ids"],
                "labels": tokenized_result["labels"],
            },
        )

        assert len(inputs) == 2
        assert len(preds) == 2

        assert inputs[0] == (
            '<|endoftext|>Convert input data to json according to JSONSchema\n[SCHEMA]{"type": "object", "properties": {"text": {"type": "string"}}}[INPUT]hello world[OUTPUT]'
        )
        assert inputs[1] == (
            '<|endoftext|>Convert input data to json according to JSONSchema\n[SCHEMA]{"type": "object", "properties": {"message": {"type": "string"}}}[INPUT]test data[OUTPUT]'
        )

        assert isinstance(preds[0], str)
        assert len(preds[0]) > 0
        assert "SCHEMA" not in preds[0]
        assert "INPUT" not in preds[0]
        assert batch["input_data"][0] not in preds[0]
        assert batch["schema"][0] not in preds[0]

        assert isinstance(preds[1], str)
        assert len(preds[1]) > 0
        assert "SCHEMA" not in preds[1]
        assert "INPUT" not in preds[1]
        assert batch["input_data"][1] not in preds[1]
        assert batch["schema"][1] not in preds[1]
