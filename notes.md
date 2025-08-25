# Any2JSON

Dataset candidates:
- https://huggingface.co/datasets/sandersaarond/Grafana-Community-Dashboards
- https://www.kaggle.com/datasets/eliasdabbas/web-server-access-logs
- https://www.kaggle.com/datasets/omduggineni/loghub-apache-log-data
- 

Idea: query infini-gram indexes for jsons
https://infini-gram.readthedocs.io/en/latest/api.html

add RDF
https://dbpedia.org/page/Resource_Description_Framework

Thoughts:
- Train/test split by schemas
- Select atypical schemas for test
- Select atypical schemas by tfidf
- Infinigram extracted non-json chunks are currently not useful

Next up:
- Work on format conversions.
    - Markdown table converter is broken, need to fix
    - Need as many converters as I can code/vibecode
- Schema variation should be moved to online data augmentation, no need to store all the variations in the database
- Need to obtain triplets you can train on asap
- And move on to at least benchmarking an llm ASAP
- And to training anything ASAP

Infinigram roughly 4 it/s

# Commands: assembling the dataset

Stage 1: loading

```
python scripts/db_tools.py init
python scripts/data_engine.py download-datasets
python scripts/data_engine.py process-dataset data/raw/wikimedia/structured-wikipedia
python scripts/data_engine.py process-dataset data/raw/ChristianAzinn/json-training
python scripts/data_engine.py process-dataset data/raw/dataunitylab/json-schema
python scripts/data_engine.py process-dataset data/raw/dataunitylab/json-schema-descriptions
python scripts/data_engine.py process-dataset data/raw/interstellarninja/json-mode-reasoning
python scripts/data_engine.py process-dataset data/raw/interstellarninja/json-mode-verifiable
python scripts/data_engine.py process-dataset data/raw/interstellarninja/json-mode-agentic-reasoning
python scripts/data_engine.py process-dataset data/raw/interstellarninja/json-schema-store-reasoning

python scripts/db_tools.py drop-duplicate-schemas
python scripts/db_tools.py drop-duplicate-documents
python scripts/db_tools.py drop-duplicate-chunks
python scripts/db_tools.py vacuum
python scripts/db_tools.py stats
```


Stage 2: infinigram, mining chunks, pandas

```
python scripts/data_engine.py get-from-infinigram --num-chunks=5000
python scripts/data_engine.py extract-json-chunks --frac-per-document=0.2 --max-depth=10 --max-chunks=12000
python scripts/data_engine.py generate-pandas-chunks --num-chunks=5000
python scripts/db_tools.py drop-duplicate-schemas
python scripts/db_tools.py drop-duplicate-chunks
python scripts/db_tools.py vacuum
python scripts/db_tools.py stats
```

Stage 3: mapping chunks to schemas, generating schemas and chunks
```
python scripts/db_tools.py cull-chunks --min-length=10 --max-length=7000

python scripts/data_engine.py map-chunks
python scripts/data_engine.py generate-schemas --num-chunks=1000
python scripts/data_engine.py map-chunks
python scripts/data_engine.py generate-schemas
python scripts/db_tools.py drop-duplicate-schemas
python scripts/data_engine.py map-chunks

python scripts/data_engine.py generate-schemas
python scripts/db_tools.py drop-duplicate-schemas
python scripts/data_engine.py map-chunks

python scripts/data_engine.py generate-chunks
python scripts/db_tools.py drop-duplicate-chunks
python scripts/db_tools.py drop-dangling-schemas
python scripts/db_tools.py vacuum
python scripts/db_tools.py stats
```

Stage 4: making format conversions
```
python scripts/data_engine.py generate-synthetic-format-conversions
python scripts/db_tools.py drop-duplicate-chunks
python scripts/db_tools.py vacuum
python scripts/db_tools.py stats
```

Stage 5: export
```
python scripts/data_engine.py assign-groups
python scripts/data_engine.py export-samples --test-size=5000
python scripts/data_engine.py export-hf-dataset --repo-id btseytlin/any2json


```

# Benchmarks

Qwen0.6B
```
# Config
{
  "hf_dataset_dir": null,
  "split": "test",
  "model_state": {
    "model_name": "Qwen/Qwen3-0.6B",
    "class_name": "QwenVLLMServer",
    "enable_thinking": true,
    "max_tokens": 1000,
    "base_url": "http://localhost:8000/v1",
    "wrapper": "QwenModel",
    "backend": "vllm_server"
  },
  "limit": null,
  "actual_samples": 4959
}
# Results
{
  "percentage_json_errors": 0.23573301068763863,
  "percentage_correct": 0.47751562815083687,
  "percentage_schema_errors": 0.10102843315184513
}
```

# GPT-5 ideas for training

🎯
I’ll outline a concise, 2025-ready recipe: model choices, data/formatting, training, and inference with schema-constrained decoding.

### High-level recommendations
- **Task framing**: Deterministic transformation with minimal reasoning. Optimize for structure-following, tokenization, and constrained decoding rather than chain-of-thought.
- **Constrained decoding is mandatory**: Compile the JsonSchema into a token-level constraint (trie/CFG) to force valid JSON, keys, enums, number/string shapes, and array/object structure.

### Model choice
- **Sweet spot (recommended)**: 0.5B–1B decoder-only code model (e.g., Qwen2.5/Qwen3 Coder 0.5–1B). Great bracket discipline, fast, cheap, strong on structure.
- **Encoder–decoder option**: ByT5/UL2-small (≈300–800M) if you prefer seq2seq and byte robustness for messy CSV/HTML/XML.
- **170M class (e.g., SmolLM 170M)**: Works on simple schemas with constraints and normalization but expect drops on long/irregular inputs and complex nesting. Use only if latency/edge constraints dominate.
- **Instruct vs base**: You do not need an instruct model. A base/code model fine-tuned on your triplets is ideal.

### Tokenization and context
- **Tokenizer**: Use byte-fallback or pure byte models (ByT5) to avoid OOV for markup and punctuation.
- **Context length**: 8–16k recommended. Use position scaling (e.g., dynamic RoPE) if available.

### Input formatting (single sequence)
- **Canonical prompt**:
  - Header: a short, stable instruction like: “Convert input to JSON conforming to the provided JsonSchema.”
  - Schema: compacted, deterministic, minified order.
  - Input: raw or lightly normalized structured text.
  - Clear markers and minimal boilerplate.
- **Example shape**:
  - [SCHEMA] {…minified schema…}
  - [INPUT] …raw csv/html/xml…
  - [OUTPUT]
- Keep keys and markers consistent across the entire dataset.

### Training recipe (SFT)
- **Objective**: Teacher forcing, loss only on target JSON (mask everything before [OUTPUT]).
- **PEFT**: QLoRA/DoRA on the small code model. r=8–32, α≈16–64, dropout 0.05–0.1. Target all attention and MLP linears.
- **Batching**: Length bucketing + sequence packing. Effective batch size 128–256 sequences.
- **Optim**: AdamW(bfloat16), lr 5e-5 to 2e-4 (lower for bigger models), wd 0.05, warmup 2–3%, cosine decay. 1–3 epochs on 200k is typical; early stop on schema-valid EM.
- **Regularization**: Small label smoothing (0.05) can help; keep temperature at 1.0 during training.
- **Curriculum**: Start with short/simple schemas, then longer/nested. Shuffle across formats (csv/html/xml) each step.
- **Data hygiene**: Dedup near-duplicates; ensure train/val/test schema and input distributions are disjoint where possible.

### Constrained decoding (critical)
- **Compile schema → token trie** at generation time to:
  - Enforce valid JSON structure and key sets.
  - Enforce enums, types, min/max items, required fields.
  - Auto-insert commas/quotes/brackets only when legal.
- **Libraries**: Use a JSON-Schema-aware constrained decoding tool (e.g., Outlines-style JSON schema engines or lm-format-enforcer equivalents). If needed, pre-compile the schema to a JSON grammar for faster tries.
- **Decoding**: Greedy (temperature 0, top_p=1). Beam size 1 is usually best with constraints.

### Preprocessing and augmentation
- **Normalization**:
  - CSV: consistent delimiter/quote escaping; trim BOM; normalize headers.
  - HTML/XML: collapse benign whitespace; optionally provide a simplified DOM serialization.
- **Augmentations**:
  - Key order shuffles in target; whitespace variations.
  - CSV delimiter and quoting variants.
  - Field reordering in source tables; inject benign noise that preserves semantics.
- **Optional hybrid**: Provide a lightweight intermediate structure (e.g., a DOM or table JSON) in addition to raw input to reduce parsing burden.

### Evaluation
- **Primary**: Schema-valid exact match rate.
- **Secondary**: Field-level F1; normalized value exact match; tree edit distance on JSON AST; latency tokens/s.
- **Robustness**: Stress on long inputs, irregular quoting, malformed tags, nested arrays/objects, enums.

### Size guidance for your 200k triplets
- **If you need >98% schema-valid EM across formats**: 0.5–1B with constraints.
- **If edge/latency is paramount and schemas are simple**: try 170–300M with constraints; expect a few points lower.
- Run two baselines: a ~300M ByT5/UL2-small and a ~0.5–0.8B code model; pick by accuracy-latency tradeoff.

### Inference and serving
- **Greedy + constraints** yields near-deterministic outputs.
- **Quantization**: 4–8 bit weights; keep KV in bf16 if memory allows.
- **Throughput**: Use continuous batching; sequence packing at inference is unnecessary with constraints, but request batching helps.

### What you likely don’t need
- **Reasoning/COT**: Not helpful.
- **RLHF**: Unnecessary; the objective is fully supervised and verifiable.
- **Instruction tuning**: Skip unless you need chatty behavior; stick to consistent seq2seq formatting.

- Final picks:
  - Start with Qwen2.5/Qwen3 Coder 0.5–1B + QLoRA + schema-constrained decoding.
  - Compare with ByT5-small/UL2-small for byte robustness on messy inputs.

- If you want, I can draft the exact training/inference scaffolding (token masking, bucketing, LoRA config, schema-to-trie converter) aligned with your repo.

- Short status: Provided a compact, actionable 2025 recipe covering model size, formatting, training, and schema-constrained decoding. No code changes yet.

- Key takeaways:
  - Use a 0.5–1B code model or 300–800M encoder–decoder; 170M is possible but riskier.
  - Constrained decoding with JsonSchema is the single biggest win.
  - Greedy decoding, loss on output only, QLoRA/DoRA, byte-fallback tokenizer, length bucketing.
  - Evaluate with schema-valid EM and field-level metrics.


# Training

Command
L4 gpu

```
!python any2json/any2json/training/train.py train --learning_rate=5e-5 --warmup-ratio 0.03 --weight-decay 0.05  --gradient-accumulation-steps 4  --eval-strategy steps  --report_to wandb --bf16 --dataloader-num-workers 10 --dataloader-prefetch-factor 8 --group-by-length True --dataloader-persistent-workers True --optim adamw_torch_fused --tf32 True --gradient-checkpointing --debug-tokens --save-strategy steps --save-total-limit 5 --num-train-epochs=4 --per-device-train-batch-size 16 --per-device-eval-batch-size 16 --auto-find-batch-size --eval-steps 500 --save-steps 500 --max-sequence-length 1024
```

# Notes on next version

Current model downsides:

1. Does not reliably change output as schema changes: removing/adding a field. Solution: augmentation, more data.
2. Not good at filtering out data and text thats not in the schema
3. Poor extrapolation to unknown formats
4. Some schemas are ambigous (e.g. the films field case). Solution: perhaps I need my own validation, more strict than the schema standard
5. Resistance to input corruption and noise could be better -> augmentation.
6. System prompt is a bunch of useless tokens.
7. Some tokens could be dropped from input, e.g. spaces, extra newlines
8. JSON -> JSON doesn't work. Model always assumes if json is the input then it must be the exact output. Solution: need to add json2json examples

Other notes:

1. Need to figure out the desired inference setup. Do I use structured outputs? Probably should
2. Try gemma3n next.
3. Switch to A100 training.
4. Try sorting by length reverse? Start from short, proceed to long. Maybe a length-based curriculum.
5. Benchmarking needs speed metrics.
6. Schemas with refs should be cut up into smaller things. I dont think we should support refs

Next actions:

1. Obtain schemas from https://github.com/SchemaStore/schemastore/tree/master/src/schemas/json and data https://github.com/SchemaStore/schemastore/blob/master/src/test/abc-supply-plan-10.1.0/abc-supply-plan.json, https://www.schemastore.org/api/json/catalog.json
4. Change handling of schemas with refs
5. Read https://json-schema.org/draft/2020-12/json-schema-core
6. Maybe integrate: https://github.com/python-jsonschema/hypothesis-jsonschema
7. Find ways to use tools to generate more data:  https://json-schema.org/tools?query=&sortBy=name&sortOrder=ascending&groupBy=toolingTypes&licenses=&languages=&drafts=&toolingTypes=&environments=&showObsolete=false&supportsBowtie=false
8. Add openapi schema objects: https://github.com/wework/json-schema-to-openapi-schema
9. Metaschemas, use schemas as json objects
10. Add XSD, XSD schemas can be converted to json schemas
11. QUICKTYPE! https://app.quicktype.io/
12. Add the latest draft link of jsonschema to the output



