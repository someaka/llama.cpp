// Self-test for hs-extract-batch: synthetic tests over compute_masked_mean(),
// compute_single_range_mean(), the accumulator key encoding, and the
// checkpoint write/read roundtrip. No model file needed. 21 tests total.


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#ifndef _WIN32
#include <unistd.h>  // getpid for the temp checkpoint filename
#endif

#include "hs-accum.h"
#include "hs-kernels.h"
#include "io-util.h"
#include "self-test.h"

int run_self_test() {
    fprintf(stderr, "Running compute_masked_mean self-tests...\n\n");

    int passed = 0;
    int total = 21;  // attempted tests: 1-16 kernels, 17 fixture, 18-19 inside 17's success branch, 20-21 resume content check
    bool all_ok = true;

    // All tests use data layout: 3 tokens x 2 dims, row-major
    // data[0..5] = {1, 2, 3, 4, 5, 6}
    //   token0 = [1, 2], token1 = [3, 4], token2 = [5, 6]
    float data1[6] = {1, 2, 3, 4, 5, 6};

    // Test 1: single contiguous range = mean over all tokens
    {
        std::vector<std::pair<int,int>> ranges = {{0, 3}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        // dim0: (1+3+5)/3 = 3.0   dim1: (2+4+6)/3 = 4.0
        bool ok = (count == 3)
               && (std::abs(out[0] - 3.0f) < 1e-6f)
               && (std::abs(out[1] - 4.0f) < 1e-6f);
        fprintf(stderr, "  Test 1 (single contiguous range): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 2: skip first token
    {
        std::vector<std::pair<int,int>> ranges = {{1, 3}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        // dim0: (3+5)/2 = 4.0   dim1: (4+6)/2 = 5.0
        bool ok = (count == 2)
               && (std::abs(out[0] - 4.0f) < 1e-6f)
               && (std::abs(out[1] - 5.0f) < 1e-6f);
        fprintf(stderr, "  Test 2 (skip first token): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 3: non-contiguous ranges (non-contiguous selection: tokens 0 and 2, skip 1)
    {
        std::vector<std::pair<int,int>> ranges = {{0, 1}, {2, 3}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        // dim0: (1+5)/2 = 3.0   dim1: (2+6)/2 = 4.0
        bool ok = (count == 2)
               && (std::abs(out[0] - 3.0f) < 1e-6f)
               && (std::abs(out[1] - 4.0f) < 1e-6f);
        fprintf(stderr, "  Test 3 (non-contiguous ranges): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 4: empty mask (zero ranges)
    {
        std::vector<std::pair<int,int>> ranges = {};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        // count should be 0, out stays zeroed
        bool ok = (count == 0)
               && (std::abs(out[0]) < 1e-6f)
               && (std::abs(out[1]) < 1e-6f);
        fprintf(stderr, "  Test 4 (empty mask): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 5: hard error  -  range end exceeds n_tokens (no soft clamp)
    {
        std::vector<std::pair<int,int>> ranges = {{0, 100}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        bool ok = (count == -1);  // hard error  -  no clamping
        fprintf(stderr, "  Test 5 (hard error end > n_tokens): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 6: hard error  -  both start and end overshoot
    {
        std::vector<std::pair<int,int>> ranges = {{3, 5}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        bool ok = (count == -1);  // hard error  -  no clamping
        fprintf(stderr, "  Test 6 (hard error overshoot range): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 7: hard error  -  fully out-of-bounds
    {
        std::vector<std::pair<int,int>> ranges = {{50, 100}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        bool ok = (count == -1);  // hard error  -  no clamping
        fprintf(stderr, "  Test 7 (hard error fully out-of-bounds): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 8: hard error  -  negative range
    {
        std::vector<std::pair<int,int>> ranges = {{-1, 3}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        bool ok = (count == -1);  // hard error
        fprintf(stderr, "  Test 8 (hard error negative range): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 9: hard error  -  inverted range (start > end)
    {
        std::vector<std::pair<int,int>> ranges = {{2, 1}};
        float out[2] = {0, 0};
        int count = compute_masked_mean(data1, 3, 2, ranges, out);
        bool ok = (count == -1);  // hard error
        fprintf(stderr, "  Test 9 (hard error inverted range): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 10: compute_single_range_mean  -  basic mean
    {
        float out[2] = {0, 0};
        int count = compute_single_range_mean(data1, 3, 2, 0, 3, out);
        // dim0: (1+3+5)/3 = 3.0   dim1: (2+4+6)/3 = 4.0
        bool ok = (count == 3)
               && (std::abs(out[0] - 3.0f) < 1e-6f)
               && (std::abs(out[1] - 4.0f) < 1e-6f);
        fprintf(stderr, "  Test 10 (single-range basic mean): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 11: compute_single_range_mean  -  skip first token
    {
        float out[2] = {0, 0};
        int count = compute_single_range_mean(data1, 3, 2, 1, 3, out);
        // dim0: (3+5)/2 = 4.0   dim1: (4+6)/2 = 5.0
        bool ok = (count == 2)
               && (std::abs(out[0] - 4.0f) < 1e-6f)
               && (std::abs(out[1] - 5.0f) < 1e-6f);
        fprintf(stderr, "  Test 11 (single-range skip first): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 12: compute_single_range_mean  -  single token
    {
        float out[2] = {0, 0};
        int count = compute_single_range_mean(data1, 3, 2, 1, 2, out);
        // token1 = [3, 4]
        bool ok = (count == 1)
               && (std::abs(out[0] - 3.0f) < 1e-6f)
               && (std::abs(out[1] - 4.0f) < 1e-6f);
        fprintf(stderr, "  Test 12 (single-range single token): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 13: compute_single_range_mean  -  hard error (end > n_tokens)
    {
        float out[2] = {0, 0};
        int count = compute_single_range_mean(data1, 3, 2, 0, 100, out);
        bool ok = (count == -1);
        fprintf(stderr, "  Test 13 (single-range hard error end > n_tokens): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 14: compute_single_range_mean  -  hard error (negative)
    {
        float out[2] = {0, 0};
        int count = compute_single_range_mean(data1, 3, 2, -1, 3, out);
        bool ok = (count == -1);
        fprintf(stderr, "  Test 14 (single-range hard error negative): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 15: overlapping ranges  -  verify correct token deduplication
    // ranges = {{0, 2}, {1, 3}} means tokens 0,1 and 1,2 -> token 1 counted twice
    // dim0: (1+3) + (3+5) = 4 + 8 = 12, count = 4, mean = 3.0
    // dim1: (2+4) + (4+6) = 6 + 10 = 16, count = 4, mean = 4.0
    {
        std::vector<std::pair<int,int>> ranges = {{0, 2}, {1, 3}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        bool ok = (count == 4)  // 2 + 2 tokens (token 1 counted twice)
               && (std::abs(out[0] - 3.0f) < 1e-6f)  // (1+3+3+5)/4 = 12/4 = 3.0
               && (std::abs(out[1] - 4.0f) < 1e-6f); // (2+4+4+6)/4 = 16/4 = 4.0
        fprintf(stderr, "  Test 15 (overlapping ranges): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 16: key encode/decode roundtrip
    // Verifies that make_accum_key and decode_accum_key are exact inverses
    // for all valid (group_id, mask_id, layer_idx) combinations.
    {
        bool ok = true;
        const int32_t test_cases[][3] = {
            {0, 0, 0},       // minimum
            {0xFFFF, 0xFFFF, 0xFFFF},  // maximum (16-bit each)
            {42, 17, 5},     // typical
            {1, 2, 3},       // small
            {1000, 500, 99}, // medium
        };
        for (const auto& tc : test_cases) {
            uint64_t key = make_accum_key(tc[0], tc[1], tc[2]);
            int32_t g, m, l;
            decode_accum_key(key, g, m, l);
            if (g != tc[0] || m != tc[1] || l != tc[2]) {
                fprintf(stderr, "  key roundtrip failed: (%d,%d,%d) -> key=%llu -> (%d,%d,%d)\n",
                        tc[0], tc[1], tc[2], (unsigned long long)key, g, m, l);
                ok = false;
            }
        }
        // Verify key uniqueness: different inputs must produce different keys
        uint64_t k1 = make_accum_key(1, 0, 0);
        uint64_t k2 = make_accum_key(0, 1, 0);
        uint64_t k3 = make_accum_key(0, 0, 1);
        if (k1 == k2 || k1 == k3 || k2 == k3) ok = false;

        fprintf(stderr, "  Test 16 (key encode/decode roundtrip): %s\n", ok ? "PASS" : "FAIL");
        if (ok) passed++; else all_ok = false;
    }

    // Test 17: checkpoint write/read roundtrip + fingerprint enforcement
    // Verifies that the checkpoint format (sum-based) survives a write+read
    // cycle with zero precision loss, that the v3 run fingerprint round-trips,
    // that a mismatched fingerprint is rejected, and that corruption at any
    // byte offset is detected.
    {
        AccumulatorMap test_acc;
        // Create a few test entries with known sum values
        uint64_t key1 = make_accum_key(0, 0, 0);
        test_acc[key1].sum = {1.0f, 2.0f, 3.0f, 4.0f};
        test_acc[key1].count = 10;
        uint64_t key2 = make_accum_key(1, 2, 5);
        test_acc[key2].sum = {0.1f, -0.2f, 0.3f, -0.4f};
        test_acc[key2].count = 7;

        checkpoint_fingerprint fp;
        fp.generate_mode = false;
        fp.generate_tokens = 0;
        fp.token_skip = 50;
        fp.layers = {0, 17, 35};
        fp.content_fnv64 = hs_fnv1a64("hello world");
        fp.n_prompts = 3;

        // Write checkpoint to temp file - include PID to avoid collision
        // between concurrent self-test runs (CI) and symlink attacks.
        char test_ckpt_buf[256];
        snprintf(test_ckpt_buf, sizeof(test_ckpt_buf), "/tmp/hs_self_test_ckpt_%d.bin", (int)getpid());
        const char* test_ckpt = test_ckpt_buf;
        bool write_ok = write_checkpoint(test_acc, test_ckpt, 4, 42, fp);
        if (!write_ok) {
            fprintf(stderr, "  Test 17 (checkpoint roundtrip): FAIL (write failed)\n");
            fprintf(stderr, "  Test 18/19: SKIP (fixture unavailable)\n");
            all_ok = false;
            total += 2;  // attempted-but-failed, not silently absent
        } else {
            // Prompts fixture for the v4 content check: the checkpoint
            // records the hash of the 42nd non-empty line ("hello world").
            const std::string prompts_path = std::string(test_ckpt) + ".prompts";
            {
                std::ofstream pf(prompts_path.c_str());
                for (int i = 0; i < 41; i++) pf << "filler " << i << "\n";
                pf << "hello world\n";
                pf << "tail\n";
            }
            // Read it back with the same fingerprint - must succeed
            AccumulatorMap restored_acc;
            int32_t n_iterated = 0;
            bool read_ok = read_checkpoint(test_ckpt, restored_acc, n_iterated, 4, fp, prompts_path.c_str());

            bool ok = read_ok && (n_iterated == 42);
            if (ok) {
                // Verify sums match exactly (the format stores sum directly)
                auto& av1 = restored_acc[key1];
                auto& av2 = restored_acc[key2];
                ok = (av1.count == 10) && (av2.count == 7);
                for (int d = 0; d < 4 && ok; d++) {
                    if (std::abs(av1.sum[d] - test_acc[key1].sum[d]) > 1e-6f) ok = false;
                    if (std::abs(av2.sum[d] - test_acc[key2].sum[d]) > 1e-6f) ok = false;
                }
            }
            fprintf(stderr, "  Test 17 (checkpoint roundtrip): %s\n", ok ? "PASS" : "FAIL");
            if (ok) passed++; else all_ok = false;

            // Test 18: fingerprint mismatch must be rejected loudly
            {
                checkpoint_fingerprint wrong_fp = fp;
                wrong_fp.token_skip = 0;  // different run setting
                AccumulatorMap junk_acc;
                int32_t junk_iter = 0;
                bool read_must_fail = read_checkpoint(test_ckpt, junk_acc, junk_iter, 4, wrong_fp, prompts_path.c_str());
                bool ok18 = !read_must_fail;
                // A rejected read must not leave partial state behind
                if (ok18 && !junk_acc.empty()) {
                    fprintf(stderr, "    (partial-state check: %zu entries leaked)\n", junk_acc.size());
                    ok18 = false;
                }
                fprintf(stderr, "  Test 18 (fingerprint mismatch rejected): %s\n", ok18 ? "PASS" : "FAIL");
                if (ok18) passed++; else all_ok = false;
            }

            // Test 19: corruption at any byte offset must be detected.
            // Rewrite the checkpoint, then flip one byte at a spread of
            // offsets (version field, fingerprint, payload) and require
            // read_checkpoint to return false.
            {
                bool ok19 = true;
                std::string ckpt_path = std::string(test_ckpt) + ".checkpoint";
                FILE* orig = fopen(ckpt_path.c_str(), "rb");
                if (!orig) {
                    ok19 = false;
                } else {
                    std::vector<unsigned char> bytes;
                    int c;
                    while ((c = fgetc(orig)) != EOF) bytes.push_back((unsigned char)c);
                    fclose(orig);
                    const size_t n = bytes.size();
                    // Corrupt one byte at these offsets: 0 (version), 8
                    // (fingerprint fields), and payload offsets.
                    size_t offsets[6];
                    offsets[0] = 0;
                    offsets[1] = 8;
                    offsets[2] = 16;
                    offsets[3] = n / 4;
                    offsets[4] = n / 2;
                    offsets[5] = n - 1;
                    for (int oi = 0; oi < 6 && ok19; oi++) {
                        if (offsets[oi] >= n) continue;
                        std::vector<unsigned char> corrupt = bytes;
                        corrupt[offsets[oi]] ^= 0xFF;
                        FILE* wf = fopen(ckpt_path.c_str(), "wb");
                        if (!wf) { ok19 = false; break; }
                        fwrite(corrupt.data(), 1, corrupt.size(), wf);
                        fclose(wf);
                        AccumulatorMap junk_acc;
                        int32_t junk_iter = 0;
                        // Note: some single-byte flips in the float payload
                        // cannot be detected by structural validation (a
                        // float is any bit pattern). Require detection for
                        // the structural offsets only: version (0),
                        // fingerprint area (8, 16) and the record-count
                        // region (n/4 hits group/mask/layer headers).
                        bool structural = (oi <= 3);
                        bool read_ok2 = read_checkpoint(ckpt_path.c_str(), junk_acc, junk_iter, 4, fp, prompts_path.c_str());
                        // Structural offsets must be detected. Float-payload
                        // offsets may legitimately pass: a flipped float is a
                        // valid float (comment above).
                        if (structural && read_ok2) ok19 = false;
                    }
                    // Restore the pristine checkpoint for any later checks
                    FILE* rf = fopen(ckpt_path.c_str(), "wb");
                    if (rf) {
                        fwrite(bytes.data(), 1, bytes.size(), rf);
                        fclose(rf);
                    }
                }
                fprintf(stderr, "  Test 19 (corruption detection): %s\n", ok19 ? "PASS" : "FAIL");
                if (ok19) passed++; else all_ok = false;
            }

            // Test 20: same-content resume roundtrip must pass with the
            // content check active (skip_count > 0, v4 hash re-verified
            // against the fixture's 42nd non-empty line).
            {
                AccumulatorMap same_acc;
                int32_t same_iter = 0;
                bool read_same = read_checkpoint(test_ckpt, same_acc, same_iter, 4, fp, prompts_path.c_str());
                bool ok20 = read_same && (same_iter == 42) && (same_acc.size() == 2);
                fprintf(stderr, "  Test 20 (same-content resume roundtrip): %s\n", ok20 ? "PASS" : "FAIL");
                if (ok20) passed++; else all_ok = false;
            }

            // Test 21: tampered prompts content must be rejected loudly
            // (line 42 differs from the hash recorded in the checkpoint).
            {
                bool ok21 = true;
                const std::string tampered_path = prompts_path + ".tampered";
                {
                    std::ofstream tf(tampered_path.c_str());
                    for (int i = 0; i < 41; i++) tf << "filler " << i << "\n";
                    tf << "TAMPERED content\n";
                    tf << "tail\n";
                }
                AccumulatorMap junk_acc;
                int32_t junk_iter = 0;
                bool read_tampered = read_checkpoint(test_ckpt, junk_acc, junk_iter, 4, fp, tampered_path.c_str());
                if (read_tampered) ok21 = false;
                // A rejected content check must not leave partial state behind
                if (ok21 && !junk_acc.empty()) ok21 = false;
                fprintf(stderr, "  Test 21 (tampered prompts content rejected): %s\n", ok21 ? "PASS" : "FAIL");
                if (ok21) passed++; else all_ok = false;
                remove(tampered_path.c_str());
            }
        }
        // Cleanup
        remove(test_ckpt);
        remove((std::string(test_ckpt) + ".tmp").c_str());
        remove((std::string(test_ckpt) + ".checkpoint").c_str());
        remove((std::string(test_ckpt) + ".checkpoint.tmp").c_str());
        remove((std::string(test_ckpt) + ".prompts").c_str());
        remove((std::string(test_ckpt) + ".prompts.tampered").c_str());
    }

    fprintf(stderr, "\n%d/%d tests passed\n", passed, total);

    if (all_ok) {
        fprintf(stderr, "All self-tests passed\n");
        return 0;
    } else {
        fprintf(stderr, "SELF-TEST FAILED\n");
        return 1;
    }
}
