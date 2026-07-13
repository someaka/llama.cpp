# A Machine Validation Plan

## Models to Download (from HuggingFace)

All models are from `bartowski` with Q4_K_M quantization:

1. **Gemma-4-E2B**: `bartowski/google_gemma-4-E2B-it-GGUF/google_gemma-4-E2B-it-Q4_K_M.gguf` (~3.5 GB)
2. **Gemma-4-E4B**: `bartowski/google_gemma-4-E4B-it-GGUF/google_gemma-4-E4B-it-Q4_K_M.gguf` (~6 GB)
3. **Qwen3.5-2B**: `bartowski/Qwen_Qwen3.5-2B-GGUF/Qwen_Qwen3.5-2B-Q4_K_M.gguf` (~1.4 GB)

## Test Data

Use the test dataset in `/home/d/Desktop/agenda/CrimsonRed/test_extraction/`:
- `prompts.txt` - 10 test prompts
- `assignments.bin` - assignments with n_embd=4096 (Gemma-4-E4B)

Note: May need to regenerate assignments.bin for each model's n_embd.

## Validation Steps

### Step 1: Compile
```bash
cd /home/d/Desktop/agenda/CrimsonRed/llama.cpp
cmake --build build --target llama-hs-extract-batch -j$(nproc)
```

### Step 2: Baseline Test (batch_size=1)
```bash
./build/bin/llama-hs-extract-batch \
  /home/d/Desktop/gguf-models/gemma4-e4b-q4_k_m.gguf \
  /home/d/Desktop/agenda/CrimsonRed/test_extraction/prompts.txt \
  all \
  /tmp/baseline.bin \
  --batch \
  --assignments /home/d/Desktop/agenda/CrimsonRed/test_extraction/assignments.bin \
  --batch-size 1 \
  --ctx-size 512
```

### Step 3: Batched Test (batch_size=4)
```bash
./build/bin/llama-hs-extract-batch \
  /home/d/Desktop/gguf-models/gemma4-e4b-q4_k_m.gguf \
  /home/d/Desktop/agenda/CrimsonRed/test_extraction/prompts.txt \
  all \
  /tmp/batched.bin \
  --batch \
  --assignments /home/d/Desktop/agenda/CrimsonRed/test_extraction/assignments.bin \
  --batch-size 4 \
  --ctx-size 512
```

### Step 4: Compare Outputs
```bash
python3 -c "
import struct
import numpy as np

def read_output(path):
    with open(path, 'rb') as f:
        # Read header
        magic = struct.unpack('<I', f.read(4))[0]
        n_groups = struct.unpack('<I', f.read(4))[0]
        n_layers = struct.unpack('<I', f.read(4))[0]
        n_embd = struct.unpack('<I', f.read(4))[0]
        
        results = []
        for g in range(n_groups):
            group_id = struct.unpack('<I', f.read(4))[0]
            n_masks = struct.unpack('<I', f.read(4))[0]
            
            for m in range(n_masks):
                mask_id = struct.unpack('<I', f.read(4))[0]
                n_layer_data = struct.unpack('<I', f.read(4))[0]
                
                for l in range(n_layer_data):
                    layer_idx = struct.unpack('<I', f.read(4))[0]
                    count = struct.unpack('<I', f.read(4))[0]
                    
                    # Read mean vector
                    mean_data = f.read(n_embd * 4)
                    mean_vec = np.frombuffer(mean_data, dtype=np.float32)
                    
                    results.append({
                        'group_id': group_id,
                        'mask_id': mask_id,
                        'layer_idx': layer_idx,
                        'count': count,
                        'mean': mean_vec
                    })
        
        return results

baseline = read_output('/tmp/baseline.bin')
batched = read_output('/tmp/batched.bin')

print(f'Baseline entries: {len(baseline)}')
print(f'Batched entries: {len(batched)}')

if len(baseline) != len(batched):
    print('ERROR: Different number of entries')
    exit(1)

max_diff = 0
for i, (b, a) in enumerate(zip(baseline, batched)):
    if b['group_id'] != a['group_id']:
        print(f'ERROR: group_id mismatch at entry {i}')
        exit(1)
    if b['mask_id'] != a['mask_id']:
        print(f'ERROR: mask_id mismatch at entry {i}')
        exit(1)
    if b['layer_idx'] != a['layer_idx']:
        print(f'ERROR: layer_idx mismatch at entry {i}')
        exit(1)
    if b['count'] != a['count']:
        print(f'ERROR: count mismatch at entry {i}')
        exit(1)
    
    diff = np.abs(b['mean'] - a['mean']).max()
    max_diff = max(max_diff, diff)
    
    if diff > 1e-5:
        print(f'ERROR: mean vector mismatch at entry {i}, max diff: {diff}')
        exit(1)

print(f'SUCCESS: All outputs match! Max difference: {max_diff:.2e}')
"
```

### Step 5: Performance Comparison
```bash
echo "=== Baseline (batch_size=1) ==="
time ./build/bin/llama-hs-extract-batch \
  /home/d/Desktop/gguf-models/gemma4-e4b-q4_k_m.gguf \
  /home/d/Desktop/agenda/CrimsonRed/test_extraction/prompts.txt \
  all \
  /tmp/perf_baseline.bin \
  --batch \
  --assignments /home/d/Desktop/agenda/CrimsonRed/test_extraction/assignments.bin \
  --batch-size 1 \
  --ctx-size 512 2>&1 | grep "prompts/sec"

echo ""
echo "=== Batched (batch_size=4) ==="
time ./build/bin/llama-hs-extract-batch \
  /home/d/Desktop/gguf-models/gemma4-e4b-q4_k_m.gguf \
  /home/d/Desktop/agenda/CrimsonRed/test_extraction/prompts.txt \
  all \
  /tmp/perf_batched.bin \
  --batch \
  --assignments /home/d/Desktop/agenda/CrimsonRed/test_extraction/assignments.bin \
  --batch-size 4 \
  --ctx-size 512 2>&1 | grep "prompts/sec"
```

## Expected Results

1. **Correctness**: Outputs should be bit-identical (max diff < 1e-6)
2. **Performance**: batch_size=4 should be ~2-3x faster than batch_size=1
   - Phase 2 (batch API): eliminates 27 of 28 GPU→CPU syncs
   - Phase 4 (multi-prompt batching): processes 4 prompts per decode call

## Troubleshooting

### "extract_hidden_states not supported"
Model doesn't support hidden state extraction. Try a different model.

### "hidden state overflow"
Missing `llama_synchronize()` call. Should be fixed in current code.

### Output mismatch
Check for floating point accumulation differences. Small diffs (<1e-5) are acceptable.
