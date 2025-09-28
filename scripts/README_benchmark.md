# Pod Benchmark Script Usage

The unified `pod_benchmark.sh` script replaces the individual `pod_benchmark_gemma.sh` and `pod_benchmark_smollm.sh` scripts.

## Usage

```bash
# Source configuration and run
source scripts/config_gemma.env && ./scripts/pod_benchmark.sh

# Or source configuration and run for SmollM
source scripts/config_smollm.env && ./scripts/pod_benchmark.sh
```

## Environment Variables

### Required
- `MODEL_ARTIFACT_ID`: WandB artifact ID (e.g., `btseytlin/any2json/model-f2yvr0zy:v6`)
  - The model name is automatically generated from this artifact ID (hash and version extracted)

### Optional
- `MODEL_TYPE`: Model type prefix for naming (default: `smollm2`)
- `BENCHMARK_LIMIT`: Number of test samples to benchmark (default: `500`)
- `RUN_STRUCTURED_OUTPUT`: Run structured output benchmark (default: `true`)
- `RUN_NO_STRUCTURED_OUTPUT`: Run non-structured output benchmark (default: `false`)
- `MODEL_PATH_OVERRIDE`: Custom model path (default: derived from artifact ID)
- `NO_SO_MODEL_PATH_OVERRIDE`: Custom model path for non-structured output benchmark

## Examples

### Gemma Configuration
```bash
export MODEL_ARTIFACT_ID=btseytlin/any2json/model-ot5q00pi:v6
export MODEL_TYPE=gemma270m
export RUN_STRUCTURED_OUTPUT=true
export RUN_NO_STRUCTURED_OUTPUT=true
export NO_SO_MODEL_PATH_OVERRIDE=/workspace/models/any2json_gemma270m:epoch1
# Model name auto-generated: gemma270m_ot5q00pi_v6 (hash and version from artifact ID)
```

### SmollM Configuration
```bash
export MODEL_ARTIFACT_ID=btseytlin/any2json/model-31cjr161:v6
export MODEL_TYPE=smollm2
export RUN_STRUCTURED_OUTPUT=true
export RUN_NO_STRUCTURED_OUTPUT=true
# Model name auto-generated: smollm2_31cjr161_v6 (hash and version from artifact ID)
```
