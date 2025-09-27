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
python scripts/db_tools.py cull-chunks --min-length=5 --max-length=8096

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
python scripts/data_engine.py generate-chunks --not-only-dangling --num-schemas 10000 
python scripts/db_tools.py drop-duplicate-chunks
python scripts/db_tools.py cull-chunks --min-length=5 --max-length=8096
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

1. Move augmentation to custom dataset __getitem__
4. Change handling of schemas with refs
5. Read https://json-schema.org/draft/2020-12/json-schema-core
6. Maybe integrate: https://github.com/python-jsonschema/hypothesis-jsonschema
7. Find ways to use tools to generate more data:  https://json-schema.org/tools?query=&sortBy=name&sortOrder=ascending&groupBy=toolingTypes&licenses=&languages=&drafts=&toolingTypes=&environments=&showObsolete=false&supportsBowtie=false
8. Add openapi schema objects: https://github.com/wework/json-schema-to-openapi-schema
9. Metaschemas, use schemas as json objects
10. Add XSD, XSD schemas can be converted to json schemas
11. QUICKTYPE! https://app.quicktype.io/
12. Add the latest draft link of jsonschema to the output


### Gemma 270m notes

1. Json2json now works. But putting json and no schema breaks.
2. Guided decoding works well if the schema is well defined and data maps to schema well. Otherwise breaks down, sometimes very bad.
3. Simple schema manipulations still dont work reliably. For example: changing a field from string to number.
4. Doesnt handle unknown types of text
5. Large sequences seem slow on cpu
6. Broken samples in dataset. E.g. in test set sample 324: schema doesnt match correct output. I am afraid I will have to recreate the dataset.
7. Most test errors seem to be with large sequences.


Ideas: 

1. Have multiple inputs and outputs per schema so the model can not guess the output from either the input or the schema 
2. Retrain smollm with chat template, high augmentations and train for long
3. Benchmark against gemili flash lite
4. Train gemma
5. Put and save input data types to meta when benchmarking so I can analyze errors by format 

### 

Submitting runpod benchmarking

```
python any2json/training/submit_runpod.py --name any2json-benchmark-a40  --script /Users/boris/Documents/any2json/scripts/pod_benchmark.sh
```

Must integrate this!

https://pypi.org/project/genson/


Finally runpod benchmarking worked

Next step: run a gemma 270m so training run on runpod for 10 epochs


Command to run gemma benchmarks on cpu:

python any2json/benchmarks/benchmark.py run --hf-dataset btseytlin/any2json --split test --model-type vllm_custom --output-dir=benchmark_results --model-kwargs='{"model_name": "/Users/boris/Documents/any2json/artifacts/any2json_gemma270m:epoch_5", "vllm_serve_args": ["--disable-sliding-window", "--max-model-len", "8000", "--max-num-batched-tokens", "8000"], "guided_json": true}' --output-dir benchmark_results/gemma270m_so --limit 1

Training Smollm and Gemma without augmentations and gemma with augmentations for both train and val to understand if its the problem

## Experiment Alpha

Smollm vs Gemma

Augs vs no augs

### Result of experiment

1. Smollm no augs: https://wandb.ai/btseytlin/any2json/runs/f2yvr0zy/overview
2. Gemma no augs: https://wandb.ai/btseytlin/any2json/runs/ot5q00pi?nw=nwuserbtseytlin
3. Gemma augs val+train: https://wandb.ai/btseytlin/any2json/runs/kcid85k4/overview

Train loss is suspiciously the same between 2 and 3. Need to confirm: does --no-augment really use no augments?

--no-augment correctly sets augment False, --augment sets True

Ok the loss is same because of no_augment_first_k_index_accesses = 2. If thats true loss should be different after 2nd epoch. But I only trained for 2 epochs! Need to change to no_augment_first_k_index_accesses=1. So I ran three expreiments with no augs

Lets measure benchmark quality

1. Artifact btseytlin/any2json/model-f2yvr0zy:v6

```
python scripts/submit_runpod.py  --name any2json-benchmark-f2yvr0zy  --script scripts/pod_benchmark_smollm.sh --template-id gmu9nenh8c --auto-terminate
```

Benchmark results:

https://wandb.ai/btseytlin/any2json-scripts/artifacts/benchmark_results/any2json-benchmark-smollm2_f2yvr0zy_v6_so/v1

2. btseytlin/any2json/model-ot5q00pi:v6

```
python scripts/submit_runpod.py  --name any2json-benchmark-ot5q00pi  --script scripts/pod_benchmark_gemma.sh --template-id gmu9nenh8c --auto-terminate
```

Benchmark results:

https://wandb.ai/btseytlin/any2json-scripts/artifacts/benchmark_results/any2json-benchmark-gemma270m_ot5q00pi_v6_so/v1

3. Is identical to 2, so no need to benchmark

We can see eval loss stops falling after 80k steps, but train loss keeps falling -> overfitting.

Lets run a training with actual augs.

#### Results summary

smollm2_f2yvr0zy_v6_so:
- Correct: 75.6% 
- Json errors: 4.2%
- Schema errors: 1.4%
- Mean diff chars: 116

gemma270m_ot5q00pi_v6_so:
- Correct: 57.8% 
- Json errors: 7.6%
- Schema errors: 9.4%
- Mean diff chars: 321

Conclusion: gemma is worse. Either I dont have enough data for it or I am not using it right. Lets stick to smollm until we finish the whole pipeline. 

Lets dive deeper into smollm.
First, benchmark without `"guided_json": true`. 


smollm2_f2yvr0zy_v6:
- Correct: 75.2% 
- Json errors: 4.0%
- Schema errors: 1.4%
- Mean diff chars: 125

I honestly dont understand how it's doing the guided json thing, lets explore

Ok so I am passing guided json true, it attaches the schema to the request payload. But vllm should freak out at the sight of my schemas which have optional fields. It does it during manual inference. Why not here? Lets check vllm server logs.

Lets try benchmarking locally and see if schemas actually get sent in the SO mode

Need to make sure format_example is applied the same way during training, inference, benchmarking

Yay, now structured outputs dont work in benchmarking just like in inference! Great!

Apparently outlines can handle optional files if the schema is like this:
```python
{'properties': {'name': {'title': 'Name', 'type': 'string'}, 'urgency': {'enum': ['high', 'medium', 'low'], 'title': 'Urgency', 'type': 'string'}, 'issue': {'title': 'Issue', 'type': 'string'}, 'reporter': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Reporter'}}, 'required': ['name', 'urgency', 'issue', 'reporter'], 'title': 'Customer', 'type': 'object'}
```

Perhaps I can convert all `["string", null]` types to `anyOf` and it will work. 

Outlines cant be used with modern vllm. Lets try xgrammar with optional types

Xgrammar compiles the schema

Even this:
```{"properties": {"name": {"title": "Name", "type": ["string", "null"]}}, "title": "Customer", "type": "object"}```

Noticed that vllm was trying to use bf16. Which explains the "logits must be fp32" error from xgrammar.

Now it suddenly works after I put ```--dtype float``` to vllm args

All previous so benches ran without so. Deleting all old benchmark results with _so postfixes to avoid confusion.

Lets rerun the benches for smollm and gemma with true so then. 


### Train with augs

Smollm: https://wandb.ai/btseytlin/any2json/runs/31cjr161/overview
Gemma: https://wandb.ai/btseytlin/any2json/runs/gbghsh7d?nw=nwuserbtseytlin



 