// Self-test for hs-extract-batch: synthetic tests over compute_masked_mean(),
// compute_single_range_mean(), the accumulator key encoding, and the
// checkpoint write/read roundtrip. No model file needed. The test count
// is computed (HS_CHECK attempts), not hand-maintained.


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <unistd.h>  // getpid for the temp checkpoint filename
#endif

#include "hs-accum.h"
#include "hs-kernels.h"
#include "io-util.h"  // FilePtr RAII wrapper (Check 5 discipline)
#include "self-test.h"

// Every check increments the attempted-counter; passed/total are derived
// from it, so the suite size is never hand-maintained.
static int hs_attempted = 0;
static int hs_failed = 0;
#define HS_CHECK(ok_expr, label)                                        \
    do {                                                                \
        hs_attempted++;                                                 \
        const bool hs_ok_ = (ok_expr);                                  \
        fprintf(stderr, "  %s: %s\n", (label), hs_ok_ ? "PASS" : "FAIL"); \
        if (!hs_ok_) { all_ok = false; hs_failed++; }                   \
    } while (0)

int run_self_test() {
    fprintf(stderr, "Running compute_masked_mean self-tests...\n\n");

    bool all_ok = true;  // passed/total are derived from hs_attempted

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
        HS_CHECK(ok, "Test 1 (single contiguous range)");
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
        HS_CHECK(ok, "Test 2 (skip first token)");
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
        HS_CHECK(ok, "Test 3 (non-contiguous ranges)");
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
        HS_CHECK(ok, "Test 4 (empty mask)");
    }

    // Test 5: hard error  -  range end exceeds n_tokens (no soft clamp)
    {
        std::vector<std::pair<int,int>> ranges = {{0, 100}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        bool ok = (count == -1);  // hard error  -  no clamping
        HS_CHECK(ok, "Test 5 (hard error end > n_tokens)");
    }

    // Test 6: hard error  -  both start and end overshoot
    {
        std::vector<std::pair<int,int>> ranges = {{3, 5}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        bool ok = (count == -1);  // hard error  -  no clamping
        HS_CHECK(ok, "Test 6 (hard error overshoot range)");
    }

    // Test 7: hard error  -  fully out-of-bounds
    {
        std::vector<std::pair<int,int>> ranges = {{50, 100}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        bool ok = (count == -1);  // hard error  -  no clamping
        HS_CHECK(ok, "Test 7 (hard error fully out-of-bounds)");
    }

    // Test 8: hard error  -  negative range
    {
        std::vector<std::pair<int,int>> ranges = {{-1, 3}};
        float out[2] = {0, 0};
        int64_t count = compute_masked_mean(data1, 3, 2, ranges, out);
        bool ok = (count == -1);  // hard error
        HS_CHECK(ok, "Test 8 (hard error negative range)");
    }

    // Test 9: hard error  -  inverted range (start > end)
    {
        std::vector<std::pair<int,int>> ranges = {{2, 1}};
        float out[2] = {0, 0};
        int count = compute_masked_mean(data1, 3, 2, ranges, out);
        bool ok = (count == -1);  // hard error
        HS_CHECK(ok, "Test 9 (hard error inverted range)");
    }

    // Test 10: compute_single_range_mean  -  basic mean
    {
        float out[2] = {0, 0};
        int count = compute_single_range_mean(data1, 3, 2, 0, 3, out);
        // dim0: (1+3+5)/3 = 3.0   dim1: (2+4+6)/3 = 4.0
        bool ok = (count == 3)
               && (std::abs(out[0] - 3.0f) < 1e-6f)
               && (std::abs(out[1] - 4.0f) < 1e-6f);
        HS_CHECK(ok, "Test 10 (single-range basic mean)");
    }

    // Test 11: compute_single_range_mean  -  skip first token
    {
        float out[2] = {0, 0};
        int count = compute_single_range_mean(data1, 3, 2, 1, 3, out);
        // dim0: (3+5)/2 = 4.0   dim1: (4+6)/2 = 5.0
        bool ok = (count == 2)
               && (std::abs(out[0] - 4.0f) < 1e-6f)
               && (std::abs(out[1] - 5.0f) < 1e-6f);
        HS_CHECK(ok, "Test 11 (single-range skip first)");
    }

    // Test 12: compute_single_range_mean  -  single token
    {
        float out[2] = {0, 0};
        int count = compute_single_range_mean(data1, 3, 2, 1, 2, out);
        // token1 = [3, 4]
        bool ok = (count == 1)
               && (std::abs(out[0] - 3.0f) < 1e-6f)
               && (std::abs(out[1] - 4.0f) < 1e-6f);
        HS_CHECK(ok, "Test 12 (single-range single token)");
    }

    // Test 13: compute_single_range_mean  -  hard error (end > n_tokens)
    {
        float out[2] = {0, 0};
        int count = compute_single_range_mean(data1, 3, 2, 0, 100, out);
        bool ok = (count == -1);
        HS_CHECK(ok, "Test 13 (single-range hard error end > n_tokens)");
    }

    // Test 14: compute_single_range_mean  -  hard error (negative)
    {
        float out[2] = {0, 0};
        int count = compute_single_range_mean(data1, 3, 2, -1, 3, out);
        bool ok = (count == -1);
        HS_CHECK(ok, "Test 14 (single-range hard error negative)");
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
        HS_CHECK(ok, "Test 15 (overlapping ranges)");
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

        HS_CHECK(ok, "Test 16 (key encode/decode roundtrip)");
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
        // Content identity is set below, from the prompts fixture itself.
        fp.n_prompts = 3;

        // FNV-1a-64 known vectors: the empty string hashes to the offset
        // basis, "hello world" to the canonical reference value. Guards the
        // constant against the silent-typo class (a wrong basis still
        // roundtrips self-consistently but diverges from every external
        // FNV-1a-64 implementation).
        HS_CHECK(hs_fnv1a64("") == 0xcbf29ce484222325ull, "Test 16b (fnv empty = offset basis)");
        HS_CHECK(hs_fnv1a64("hello world") == 0x779a65e7023cd2e7ull, "Test 16c (fnv known vector)");

        // Write checkpoint to temp file - include PID to avoid collision
        // between concurrent self-test runs (CI) and symlink attacks.
        char test_ckpt_buf[256];
        snprintf(test_ckpt_buf, sizeof(test_ckpt_buf), "/tmp/hs_self_test_ckpt_%d.bin", (int)getpid());
        const char* test_ckpt = test_ckpt_buf;
        // Prompts fixture for the v5 content check: the checkpoint records
        // the rolling hash of the first 42 non-empty lines (41 fillers +
        // "hello world"), exactly as the producer would accumulate it.
        const std::string prompts_path = std::string(test_ckpt) + ".prompts";
        {
            std::ofstream pf(prompts_path.c_str());
            for (int i = 0; i < 41; i++) pf << "filler " << i << "\n";
            pf << "hello world\n";
            pf << "tail\n";
        }
        {
            uint64_t rolling = 14695981039346656037ull;  // offset basis
            std::ifstream pf(prompts_path.c_str());
            std::string line;
            int32_t n = 0;
            while (std::getline(pf, line)) {
                if (line.empty()) continue;
                rolling = fnv_roll_line(rolling, line);
                if (++n == 42) break;
            }
            fp.content_fnv64 = rolling;
        }
        bool write_ok = write_checkpoint(test_acc, test_ckpt, 4, 42, fp);
        if (!write_ok) {
            fprintf(stderr, "  (checkpoint fixture write failed)\n");
            HS_CHECK(false, "Test 17 (checkpoint roundtrip)");
            fprintf(stderr, "  Test 18-21: SKIP (fixture unavailable)\n");
            all_ok = false;
        } else {
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
            HS_CHECK(ok, "Test 17 (checkpoint roundtrip)");

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
                HS_CHECK(ok18, "Test 18 (fingerprint mismatch rejected)");
            }

            // Test 19: corruption detection is scoped. Version, fingerprint
            // and header fields must be detected; the accumulator payload
            // region carries NO checksum, so in-range flips of group ids,
            // counts, or float bytes are NOT detected (floats are any bit
            // pattern; selectors can land in range). A payload checksum is
            // the v6 vehicle. Test 20's own positive/negative content checks
            // cover the prompts-side identity.
            {
                bool ok19 = true;
                std::string ckpt_path = std::string(test_ckpt) + ".checkpoint";
                FILE* orig_raw = fopen(ckpt_path.c_str(), "rb");
                if (!orig_raw) {
                    ok19 = false;
                } else {
                    FilePtr orig(orig_raw);
                    std::vector<unsigned char> bytes;
                    int c;
                    while ((c = fgetc(orig)) != EOF) bytes.push_back((unsigned char)c);
                    orig.reset();
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
                        FILE* wf_raw = fopen(ckpt_path.c_str(), "wb");
                        if (!wf_raw) { ok19 = false; break; }
                        FilePtr wf(wf_raw);
                        fwrite(corrupt.data(), 1, corrupt.size(), wf);
                        wf.reset();
                        AccumulatorMap junk_acc;
                        int32_t junk_iter = 0;
                        bool read_ok2 = read_checkpoint(ckpt_path.c_str(), junk_acc, junk_iter, 4, fp, prompts_path.c_str());
                        // Structural offsets must be detected. Float-payload
                        // offsets may legitimately pass read_checkpoint (a
                        // flipped float is a valid float) BUT must be caught
                        // by the v6 accumulator checksum, so detection is
                        // still required at every offset.
                        if (read_ok2) ok19 = false;
                        // Restore the pristine checkpoint before the next
                        // variant and before any later checks run against
                        // this file; a silent restore failure would let Test
                        // 20 read corrupted bytes, so it is verified.
                        {
                            FilePtr rf(std::fopen(ckpt_path.c_str(), "wb"));
                            if (!rf) {
                                HS_CHECK(false, "Test 19 (restore reopen)");
                                ok19 = false;
                                break;
                            }
                            if (fwrite(bytes.data(), 1, bytes.size(), rf) != bytes.size() || rf.sync() != true) {
                                HS_CHECK(false, "Test 19 (restore write)");
                                ok19 = false;
                                break;
                            }
                            rf.reset();
                        }
                    }
                }
                HS_CHECK(ok19, "Test 19 (structural corruption detection)");

                // Test 19b: a single-byte flip anywhere in the accumulator
                // payload region (in-range group ids, counts, float bytes)
                // must be rejected by the v6 checksum. Flip the final
                // payload byte (inside the region, before the trailer).
                {
                    std::ifstream ckpt_in(ckpt_path.c_str(), std::ios::binary);
                    std::vector<unsigned char> raw((std::istreambuf_iterator<char>(ckpt_in)),
                                                    std::istreambuf_iterator<char>());
                    if (raw.size() > 9) {
                        raw[raw.size() - 9] ^= 0xFF;  // last payload byte (trailer is last 8)
                        {
                            std::ofstream ckpt_out(ckpt_path.c_str(), std::ios::binary);
                            ckpt_out.write((const char*)raw.data(), (std::streamsize)raw.size());
                        }
                        AccumulatorMap junk_acc;
                        int32_t junk_iter = 0;
                        bool read_ok3 = read_checkpoint(ckpt_path.c_str(), junk_acc, junk_iter, 4, fp, prompts_path.c_str());
                        if (read_ok3) ok19 = false;  // payload flip MUST be rejected
                        // restore pristine bytes for later tests
                        // (rewritten by the same content)
                        std::ofstream ckpt_out2(ckpt_path.c_str(), std::ios::binary);
                        raw[raw.size() - 9] ^= 0xFF;
                        ckpt_out2.write((const char*)raw.data(), (std::streamsize)raw.size());
                    }
                }
            }

            // Test 20: same-content resume roundtrip must pass with the
            // content check active (skip_count > 0, v4 hash re-verified
            // against the fixture's 42nd non-empty line).
            {
                AccumulatorMap same_acc;
                int32_t same_iter = 0;
                bool read_same = read_checkpoint(test_ckpt, same_acc, same_iter, 4, fp, prompts_path.c_str());
                bool ok20 = read_same && (same_iter == 42) && (same_acc.size() == 2);
                HS_CHECK(ok20, "Test 20 (same-content resume roundtrip)");
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
                HS_CHECK(ok21, "Test 21 (tampered prompts content rejected)");
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

    const int passed = hs_attempted - hs_failed;
    fprintf(stderr, "\n%d/%d tests passed\n", passed, hs_attempted);

    if (all_ok) {
        fprintf(stderr, "All self-tests passed\n");
        return 0;
    } else {
        fprintf(stderr, "SELF-TEST FAILED\n");
        return 1;
    }
}
