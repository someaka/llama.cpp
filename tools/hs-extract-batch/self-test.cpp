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
#include "io-util.h"  // FilePtr RAII wrapper (raw fclose is confined to this wrapper)
#include "assignments-io.h"  // T3: CRD1 reader + W4 exact-EOF probe
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
            fprintf(stderr, "  Tests 18-21 unavailable - failing\n");
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

            // Test 19: corruption detection at every offset. Version,
            // fingerprint and header fields are caught by field validation;
            // accumulator payload flips (group ids, counts, float bytes) are
            // caught by the v6 region checksum. Detection is required at
            // every sampled offset. Test 20/21 cover the prompts-side
            // content identity.
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
                        HS_CHECK(!read_ok3, "Test 19b (v6 payload-flip rejected)");
                        // restore pristine bytes for later tests
                        // (rewritten by the same content)
                        std::ofstream ckpt_out2(ckpt_path.c_str(), std::ios::binary);
                        raw[raw.size() - 9] ^= 0xFF;
                        ckpt_out2.write((const char*)raw.data(), (std::streamsize)raw.size());
                    }
                }
            }

            // Test 20: same-content resume roundtrip must pass with the
            // content check active (skip_count > 0; the v6 rolling content
            // hash re-derives cleanly from the fixture prompts — the
            // tampered variant is rejected by Test 21).
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

        // Test 22 (T1): CRD2 OUTPUT format round-trip. The production writer
        // (write_batch_output: mean = sum/count) is what every downstream
        // consumer reads; before this test it had zero coverage and a
        // regression in the division/count/buffer would ship green. Verifies
        // the header fields, the sorted (group, mask, layer) emission order,
        // the per-record count, the DIVIDED mean values, and exact EOF.
        {
            AccumulatorMap out_acc;
            uint64_t ok1_key = make_accum_key(0, 0, 0);
            out_acc[ok1_key].sum = {5.0f, 10.0f, 15.0f, 20.0f};  // count=5 -> mean {1,2,3,4}
            out_acc[ok1_key].count = 5;
            uint64_t ok2_key = make_accum_key(2, 1, 3);
            out_acc[ok2_key].sum = {2.0f, 4.0f, 6.0f, 8.0f};     // count=2 -> mean {1,2,3,4}
            out_acc[ok2_key].count = 2;

            const std::string out_path = std::string(test_ckpt) + ".out";
            bool ok22 = write_batch_output(out_acc, out_path.c_str(), 4);
            if (ok22) {
                FILE* rf = fopen(out_path.c_str(), "rb");
                ok22 = (rf != nullptr);
                if (ok22) {
                    int32_t magic = 0, n_groups = 0, n_layers = 0, n_embd = 0;
                    ok22 = fread(&magic, sizeof(int32_t), 1, rf) == 1 && magic == OUTPUT_MAGIC
                        && fread(&n_groups, sizeof(int32_t), 1, rf) == 1 && n_groups == 2
                        && fread(&n_layers, sizeof(int32_t), 1, rf) == 1 && n_layers == 4   // max_layer(3)+1
                        && fread(&n_embd, sizeof(int32_t), 1, rf) == 1 && n_embd == 4;
                    // Sorted flat-key order: (0,0,0) then (2,1,3)
                    const int32_t expect_gid[2]  = {0, 2};
                    const int32_t expect_mid[2]  = {0, 1};
                    const int32_t expect_lid[2]  = {0, 3};
                    const int32_t expect_cnt[2]  = {5, 2};
                    for (int g = 0; g < 2 && ok22; g++) {
                        int32_t gid = 0, n_masks = 0, mid = 0, n_ld = 0, lid = 0, cnt = 0;
                        float mean[4] = {0, 0, 0, 0};
                        ok22 = fread(&gid, sizeof(int32_t), 1, rf) == 1 && gid == expect_gid[g]
                            && fread(&n_masks, sizeof(int32_t), 1, rf) == 1 && n_masks == 1
                            && fread(&mid, sizeof(int32_t), 1, rf) == 1 && mid == expect_mid[g]
                            && fread(&n_ld, sizeof(int32_t), 1, rf) == 1 && n_ld == 1
                            && fread(&lid, sizeof(int32_t), 1, rf) == 1 && lid == expect_lid[g]
                            && fread(&cnt, sizeof(int32_t), 1, rf) == 1 && cnt == expect_cnt[g]
                            && fread(mean, sizeof(float), 4, rf) == 4;
                        if (ok22) {
                            // the divided mean, not the raw sum
                            for (int d = 0; d < 4; d++) {
                                if (std::abs(mean[d] - (float)(d + 1)) > 1e-6f) { ok22 = false; break; }
                            }
                        }
                    }
                    if (ok22) {  // exact EOF: no trailing bytes
                        int32_t probe = 0;
                        ok22 = fread(&probe, 1, 1, rf) == 0 && feof(rf) != 0;
                    }
                    fclose(rf);
                }
            }
            remove(out_path.c_str());
            remove((out_path + ".tmp").c_str());
            HS_CHECK(ok22, "Test 22 (CRD2 output roundtrip, mean = sum/count)");
        }

        // Test 23 (T3): CRD1 assignments reader - the production input path
        // with its own validation had no tests at all. Pins the documented
        // eof/ok/error status contract (assignments-io.h), the group-name
        // table, both mask types, and the W4 exact-EOF probe.
        {
            const std::string assign_path = std::string(test_ckpt) + ".assign";
            bool ok23 = true;
            {
                FILE* wf = fopen(assign_path.c_str(), "wb");
                ok23 = (wf != nullptr);
                if (ok23) {
                    const int32_t magic = ASSIGNMENTS_MAGIC;
                    const int32_t n_prompts = 2, n_embd_hdr = 4, n_groups = 1;
                    const int32_t name_len = 5;
                    fwrite(&magic, sizeof(int32_t), 1, wf);
                    fwrite(&n_prompts, sizeof(int32_t), 1, wf);
                    fwrite(&n_embd_hdr, sizeof(int32_t), 1, wf);
                    fwrite(&n_groups, sizeof(int32_t), 1, wf);
                    fwrite(&name_len, sizeof(int32_t), 1, wf);
                    fwrite("alpha", 1, 5, wf);
                    // prompt 0: one simple_skip assignment (mask_type 0)
                    int32_t n_assign = 1, gid = 0, mid = 0, mtype0 = 0, skip = 3;
                    fwrite(&n_assign, sizeof(int32_t), 1, wf);
                    fwrite(&gid, sizeof(int32_t), 1, wf);
                    fwrite(&mid, sizeof(int32_t), 1, wf);
                    fwrite(&mtype0, sizeof(int32_t), 1, wf);
                    fwrite(&skip, sizeof(int32_t), 1, wf);
                    // prompt 1: one explicit_ranges assignment (mask_type 1)
                    int32_t mtype1 = 1, n_ranges = 2, s0 = 0, e0 = 2, s1 = 5, e1 = 7;
                    fwrite(&n_assign, sizeof(int32_t), 1, wf);
                    fwrite(&gid, sizeof(int32_t), 1, wf);
                    fwrite(&mid, sizeof(int32_t), 1, wf);
                    fwrite(&mtype1, sizeof(int32_t), 1, wf);
                    fwrite(&n_ranges, sizeof(int32_t), 1, wf);
                    fwrite(&s0, sizeof(int32_t), 1, wf);
                    fwrite(&e0, sizeof(int32_t), 1, wf);
                    fwrite(&s1, sizeof(int32_t), 1, wf);
                    fwrite(&e1, sizeof(int32_t), 1, wf);
                    fclose(wf);
                }
            }
            if (ok23) {
                FILE* rf = fopen(assign_path.c_str(), "rb");
                ok23 = (rf != nullptr);
                if (ok23) {
                    int32_t n_prompts_hdr = 0, n_embd_hdr = 0;
                    GroupTable gt;
                    ok23 = read_assignments_header(rf, n_prompts_hdr, n_embd_hdr, gt)
                        && n_prompts_hdr == 2 && n_embd_hdr == 4
                        && gt.n_groups == 1 && gt.names.size() == 1 && gt.names[0] == "alpha";
                    if (ok23) {
                        auto r0 = read_prompt_assignments(rf);
                        ok23 = r0.status == AssignmentReadStatus::ok
                            && r0.assignments.size() == 1
                            && r0.assignments[0].mask_type == 0
                            && r0.assignments[0].skip == 3;
                    }
                    if (ok23) {
                        auto r1 = read_prompt_assignments(rf);
                        ok23 = r1.status == AssignmentReadStatus::ok
                            && r1.assignments.size() == 1
                            && r1.assignments[0].mask_type == 1
                            && r1.assignments[0].ranges.size() == 2
                            && r1.assignments[0].ranges[0] == std::make_pair(0, 2)
                            && r1.assignments[0].ranges[1] == std::make_pair(5, 7);
                    }
                    if (ok23) {
                        ok23 = read_assignments_exact_eof(rf);
                    }
                    if (ok23) {
                        auto r2 = read_prompt_assignments(rf);
                        ok23 = r2.status == AssignmentReadStatus::eof;
                    }
                    fclose(rf);
                }
            }
            HS_CHECK(ok23, "Test 23 (CRD1 reader: status contract, ranges, exact EOF)");

            // Test 23b (B4): a negative skip must be rejected at the parse
            // site, not surface later as a compute_masked_mean failure.
            const std::string bad_path = assign_path + ".badskip";
            {
                FILE* wf = fopen(bad_path.c_str(), "wb");
                if (wf) {
                    const int32_t magic = ASSIGNMENTS_MAGIC;
                    const int32_t n_prompts = 1, n_embd_hdr = 4, n_groups = 0;
                    const int32_t n_assign = 1, gid = 0, mid = 0, mtype = 0, skip = -3;
                    fwrite(&magic, sizeof(int32_t), 1, wf);
                    fwrite(&n_prompts, sizeof(int32_t), 1, wf);
                    fwrite(&n_embd_hdr, sizeof(int32_t), 1, wf);
                    fwrite(&n_groups, sizeof(int32_t), 1, wf);
                    fwrite(&n_assign, sizeof(int32_t), 1, wf);
                    fwrite(&gid, sizeof(int32_t), 1, wf);
                    fwrite(&mid, sizeof(int32_t), 1, wf);
                    fwrite(&mtype, sizeof(int32_t), 1, wf);
                    fwrite(&skip, sizeof(int32_t), 1, wf);
                    fclose(wf);
                }
                bool ok23b = false;
                FILE* rf = fopen(bad_path.c_str(), "rb");
                if (rf) {
                    int32_t hp = 0, he = 0;
                    GroupTable g2;
                    if (read_assignments_header(rf, hp, he, g2)) {
                        auto res = read_prompt_assignments(rf);
                        ok23b = res.status == AssignmentReadStatus::error;
                    }
                    fclose(rf);
                }
                remove(bad_path.c_str());
                HS_CHECK(ok23b, "Test 23b (negative skip rejected at parse site)");
            }
            remove(assign_path.c_str());
        }

        // Test 24 (T4): legacy v1 checkpoint restore. The mean*count
        // reconstruction branch (the only reader for pre-2026-07 checkpoints)
        // was never exercised; hand-crafts a minimal v1 file and verifies the
        // restored sums within the documented ~1 ULP/dim budget.
        {
            // read_checkpoint() opens <output_path> + ".checkpoint": the fixture
            // must live at that derived name.
            const std::string v1_base = std::string(test_ckpt) + ".v1run";
            const std::string v1_path = v1_base + ".checkpoint";
            bool ok24 = false;
            FILE* wf = fopen(v1_path.c_str(), "wb");
            if (wf) {
                const int32_t version = 1;
                const int32_t n_iter_v1 = 5;
                const int32_t magic = OUTPUT_MAGIC;
                const int32_t n_groups = 1, n_layers_hdr = 1, n_embd_v1 = 4;
                const int32_t group_id = 1, n_masks = 1, mask_id = 0, n_layers_data = 1;
                const int32_t layer_idx = 2;
                const uint64_t key = make_accum_key(group_id, mask_id, layer_idx);
                const int32_t count = 2;
                const float mean[4] = {1.0f, 2.0f, 3.0f, 4.0f};
                fwrite(&version, sizeof(int32_t), 1, wf);
                fwrite(&n_iter_v1, sizeof(int32_t), 1, wf);
                fwrite(&magic, sizeof(int32_t), 1, wf);
                fwrite(&n_groups, sizeof(int32_t), 1, wf);
                fwrite(&n_layers_hdr, sizeof(int32_t), 1, wf);
                fwrite(&n_embd_v1, sizeof(int32_t), 1, wf);
                fwrite(&group_id, sizeof(int32_t), 1, wf);
                fwrite(&n_masks, sizeof(int32_t), 1, wf);
                fwrite(&mask_id, sizeof(int32_t), 1, wf);
                fwrite(&n_layers_data, sizeof(int32_t), 1, wf);
                // NOTE: no key on disk - the reader reconstructs the flat key
                // from (group_id, mask_id, layer_idx).
                fwrite(&layer_idx, sizeof(int32_t), 1, wf);
                fwrite(&count, sizeof(int32_t), 1, wf);
                fwrite(mean, sizeof(float), 4, wf);
                fclose(wf);

                AccumulatorMap v1_acc;
                int32_t v1_iter = 0;
                checkpoint_fingerprint v1_fp = fp;  // v1 carries no fingerprint; not compared
                bool read_v1 = read_checkpoint(v1_base.c_str(), v1_acc, v1_iter, 4, v1_fp, nullptr);
                ok24 = read_v1 && (v1_iter == 5);
                if (ok24) {
                    auto it = v1_acc.find(key);
                    ok24 = it != v1_acc.end();
                    if (ok24) {
                        ok24 = it->second.count == 2;
                        const float expect_sum[4] = {2.0f, 4.0f, 6.0f, 8.0f};
                        for (int d = 0; d < 4 && ok24; d++) {
                            if (std::abs(it->second.sum[d] - expect_sum[d]) > 1e-4f) ok24 = false;
                        }
                    }
                }
                remove(v1_path.c_str());
                remove(v1_base.c_str());
            }
            HS_CHECK(ok24, "Test 24 (legacy v1 checkpoint restore, mean*count)");
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
