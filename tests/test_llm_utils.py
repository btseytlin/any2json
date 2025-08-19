import json
import pytest
import torch
from transformers import AutoTokenizer

from any2json.training.utils import (
    format_example,
    pad_to_multiple,
    build_train_sequence,
    build_tokenize_fn,
    CausalLMDataCollator,
    resolve_pad_id,
    build_gen_toks,
    build_tokenized_length_filter_fn,
    ids_to_token_str,
    encode_prompt_ids,
    encode_target_ids,
)


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")


def test_format_example() -> None:
    s = format_example("input", "schema")
    expected = "[SCHEMA]schema[INPUT]input[OUTPUT]"
    assert s == expected

    s = format_example("input", "schema", "output")
    expected = "[SCHEMA]schema[INPUT]input[OUTPUT]output"
    assert s == expected


class TestPadToMultiple:
    def test_pad_to_multiple_no_pad(self) -> None:
        ids, attn = pad_to_multiple(
            ids=[9, 8, 7],
            multiple=3,
            pad_id=0,
        )
        assert ids == [9, 8, 7]
        assert attn == [1, 1, 1]

    def test_pad_to_multiple_base(self) -> None:
        ids, attn = pad_to_multiple(
            ids=[9, 8, 7],
            multiple=4,
            pad_id=0,
        )
        assert ids == [9, 8, 7, 0]
        assert attn == [1, 1, 1, 0]

    def test_pad_to_multiple_larger_input(self) -> None:
        ids, attn = pad_to_multiple(
            ids=[9, 8, 7, 6, 5, 4],
            multiple=4,
            pad_id=0,
        )
        assert ids == [9, 8, 7, 6, 5, 4, 0, 0]
        assert attn == [1, 1, 1, 1, 1, 1, 0, 0]

    def test_pad_to_multiple_eight(self) -> None:
        ids, attn = pad_to_multiple(
            ids=[9, 8, 7],
            multiple=8,
            pad_id=0,
        )
        assert ids == [9, 8, 7, 0, 0, 0, 0, 0]
        assert attn == [1, 1, 1, 0, 0, 0, 0, 0]


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
            == f"{tokenizer.bos_token}[SCHEMA]{batch['schema'][0]}[INPUT]hello world[OUTPUT]{batch['output'][0]}{tokenizer.eos_token}"
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
                == f"{tokenizer.bos_token}[SCHEMA]{batch['schema'][i]}[INPUT]{batch['input_data'][i]}[OUTPUT]{batch['output'][i]}{tokenizer.eos_token}"
            )

            target_ids = [l for l in out["labels"][i] if l != -100]
            expected_target = f"{batch['output'][i]}{tokenizer.eos_token}"
            assert tokenizer.decode(target_ids) == expected_target


def test_causal_lm_data_collator_padding(tokenizer) -> None:
    coll = CausalLMDataCollator(tokenizer, pad_to_multiple_of=4)
    features = [
        {"input_ids": [3, 4, 5], "labels": [-100, -100, 7], "length": 3},
        {"input_ids": [6, 7, 8, 9, 10], "labels": [-100, -100, 1, 2, 3], "length": 5},
    ]
    out = coll(features)
    assert out["input_ids"].shape == torch.Size([2, 8])
    assert out["attention_mask"][0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    pad_id = resolve_pad_id(tokenizer)
    assert out["input_ids"][0, 3:].tolist() == [pad_id] * 5
    assert out["labels"][0, 3:].tolist() == [-100] * 5


def test_resolve_pad_id_variants(tokenizer) -> None:
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


def test_build_gen_toks(tokenizer) -> None:
    device = torch.device("cpu")
    pad_id = resolve_pad_id(tokenizer)
    toks, padded, attn = build_gen_toks(device, [1, 2, 3], 4, pad_id)
    assert padded == [1, 2, 3, pad_id]
    assert attn == [1, 1, 1, 0]
    assert toks["input_ids"].shape == torch.Size([1, 4])


def test_build_tokenized_length_filter_fn() -> None:
    pred = build_tokenized_length_filter_fn(5, 2)
    batch = {
        "input_ids": [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
        "labels": [[-100, -100, 7, 8, -100], [-100, 7, 8, 9, -100]],
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
    b = tokenizer(format_example("in", "sch"), add_special_tokens=False)["input_ids"]
    assert a == b

    c = encode_target_ids(tokenizer, "out")
    d = tokenizer("out", add_special_tokens=False)["input_ids"]
    assert c == d


def test_collator_fallback_pad_id(tokenizer) -> None:
    class NoPad:
        def __init__(self) -> None:
            self.pad_token_id = None
            self.eos_token_id = 9
            self.unk_token_id = 3

    coll = CausalLMDataCollator(NoPad(), pad_to_multiple_of=4)
    out = coll(
        [
            {"input_ids": [1, 2, 3], "labels": [4, 5, 6]},
        ]
    )
    assert out["input_ids"][0, 3:].tolist() == [9]
    assert out["labels"][0, 3:].tolist() == [-100]


def test_real_tokenizer_properties(tokenizer) -> None:
    assert hasattr(tokenizer, "eos_token_id")
    assert hasattr(tokenizer, "model_max_length")
    assert tokenizer.eos_token_id is not None

    text = "Hello world"
    enc = tokenizer(text, add_special_tokens=False)
    assert "input_ids" in enc
    assert isinstance(enc["input_ids"], list)

    decoded = tokenizer.decode(enc["input_ids"])
    assert isinstance(decoded, str)


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
