# Handoff: D → A — 2026-07-19 (Session 4 — Vulkan Hidden States Fix)

## Status: Demo LIVE on D (Vulkan), 5/5 emotion accuracy, critical bug fixed

## Critical Bug Fixed: Hidden State Buffer Offset Units (C1)

**Commit:** `32ad45fd3` on someaka/llama.cpp main

### The Bug

The `/hidden-states` endpoint returned **all zeros** for any multi-token request
(n_tokens > 1). This made the CrimsonRed demo completely broken on D — all
emotion scores were inverted because scoring ran against `-global_mean` instead
of actual hidden states.

### Root Cause

Two bugs in `src/llama-context.cpp` `decode()`:

1. **Buffer offset units mismatch (root cause):**
   ```cpp
   // BEFORE (buggy):
   const size_t dst_offset = (size_t)il * n_tokens_all * n_embd_out;  // floats!
   float * dst = hidden_state_buf.data() + dst_offset / sizeof(float);  // /4!
   ```
   `dst_offset` was computed in **floats** but used as a **byte offset** in
   the overflow check and then divided by `sizeof(float)` for pointer
   arithmetic. For single-token requests (n_tokens=1), the misalignment
   happened to fall within the buffer. For multi-token requests, data was
   written to the wrong layer slots, producing all-zeros for layers > 0.

2. **Vulkan async tensor read silent failure (secondary):**
   `ggml_backend_tensor_get_async()` calls the Vulkan backend's
   `ggml_vk_buffer_read_2d_async()` with `sync_staging=false`. When the
   destination memory is not pinned AND `sync_staging=false`, the Vulkan
   function returns `false` **without copying any data**. The caller in
   `llama-context.cpp` never checked this return value, so the buffer
   stayed zero-initialized.

### The Fix

```cpp
// AFTER (fixed):
const size_t dst_offset = (size_t)il * n_tokens_all * n_embd_out * sizeof(float)
                        + (size_t)n_tokens_prev * n_embd_out * sizeof(float);
float * dst = hidden_state_buf.data() + dst_offset / sizeof(float);
ggml_backend_tensor_get(t, dst, 0, copy_size);  // sync, not async
```

1. `dst_offset` now in **bytes** (multiplied by `sizeof(float)`) for
   consistency with `copy_size` and the overflow guard.
2. Switched to synchronous `ggml_backend_tensor_get()` to avoid the Vulkan
   silent-failure path. Performance impact is negligible since we
   synchronize immediately after the copy anyway.

### Verification

| Check | Result |
|-------|--------|
| `/hidden-states` 9-token request | 18432/18432 floats non-zero ✓ |
| `/hidden-states` 1-token request | 2048/2048 floats non-zero ✓ |
| `llama-hs-extract-batch --self-test` | 17/17 PASS ✓ |
| CrimsonRed `/api/analyze` 5/5 test | All emotion families correct ✓ |

#### 5/5 Canonical Emotion Test (D, Vulkan, Qwen3.5-2B Q4_K_M):

| Input | Top Emotion | Score |
|-------|-------------|-------|
| "I am so happy to see you today!" | optimistic | 7.513 |
| "I am furious and outraged..." | hateful | 9.534 |
| "I feel so lonely and heartbroken..." | heartbroken | 11.455 |
| "I am terrified and panicked..." | panicked | 7.564 |
| "I feel peaceful and serene..." | fulfilled | 7.375 |

Each top emotion matches the expected emotion family. The demo is fully
operational on Vulkan.

### Why A Didn't Catch This

CUDA's async copy path succeeds with pinned memory staging, so the bug
was invisible on the 3090. The Vulkan backend has a different async path
that fails silently when the destination is not pinned memory. This only
manifests on D (Vulkan iGPU with non-pinned host memory).

### Impact

This bug has been present since the hidden-states feature was added. Every
D-machine demo run since then was producing inverted scores. The July 19
handoff ("Demo LIVE on D, scoring correct") was wrong — the scores were
inverted, not correct. This fix makes the D demo actually work.

## Open Items for A

1. **Verify CUDA still works** — the offset fix is backend-independent and
   should not affect CUDA, but please run the 5/5 test on the 3090 to
   confirm.

2. **Consider upstreaming the offset fix** — this is a real bug in the
   hidden-states feature. The `sizeof(float)` correction is necessary
   regardless of backend.

3. **Vulkan async path** — the silent failure in
   `ggml_vk_buffer_read_2d_async` when `sync_staging=false` is an upstream
   Vulkan backend issue. The workaround (use sync read) is fine for
   API-driven extraction. For batch extraction throughput, this may need
   a proper fix in the Vulkan backend.

## Git State

- **llama.cpp:** `32ad45fd3` on main, clean, pushed
- **CrimsonRed:** `d97dbde` on main, clean, pushed (layer=12→10 fix from earlier this session)
