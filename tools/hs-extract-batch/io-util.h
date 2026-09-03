// Internal shared header for hs-extract-batch: RAII FILE wrapper, checked
// writes, and durable/atomic I/O helpers (checkpoint read/write, atomic
// output finalization).

#pragma once

#include <cstdio>
#include <cstdint>
#ifndef _WIN32
#include <unistd.h>  // fsync for durability before rename (POSIX)
#include <fcntl.h>   // open/O_DIRECTORY for parent-dir fsync after rename
#include <cerrno>
#endif
#include <string>
#include <vector>

#include "hs-accum.h"

// -- RAII Wrappers for Automatic Resource Cleanup -----------------------

struct FilePtr {
    FILE* fp;
    FilePtr() : fp(nullptr) {}
    FilePtr(FILE* f) : fp(f) {}
    ~FilePtr() { if (fp) fclose(fp); }
    FilePtr(const FilePtr&) = delete;
    FilePtr& operator=(const FilePtr&) = delete;
    // Move semantics (needed for records_file pattern)
    FilePtr(FilePtr&& other) noexcept : fp(other.fp) { other.fp = nullptr; }
    FilePtr& operator=(FilePtr&& other) noexcept {
        if (this != &other) {
            if (fp) fclose(fp);
            fp = other.fp;
            other.fp = nullptr;
        }
        return *this;
    }
    operator FILE*() const { return fp; }
    explicit operator bool() const { return fp != nullptr; }
    void reset() { if (fp) { fclose(fp); fp = nullptr; } }
#ifndef _WIN32
    // Flush + fsync to guarantee data reaches durable storage before a rename.
    // fflush alone only pushes libc buffers to the kernel; fsync forces the
    // kernel to write them to disk. Without this, a power loss between rename()
    // and kernel writeback can leave the renamed file containing zeros.
    // Returns false if either step fails (disk full, I/O error); the caller
    // MUST check this; ignoring it defeats the entire atomic-rename guarantee.
    bool sync() {
        if (!fp) return false;
        if (fflush(fp) != 0) return false;
        if (fsync(fileno(fp)) != 0) return false;
        return true;
    }
#else
    bool sync() {
        if (!fp) return false;
        if (fflush(fp) != 0) return false;
        if (_commit(_fileno(fp)) != 0) return false;
        return true;
    }
#endif
};

// fsync the parent directory of path after an atomic rename, so the rename
// itself survives a power cut (fsync of the data file alone can lose the
// rename). Returns false on failure; callers must propagate (a failed
// directory fsync means the rename may not be durable -- never continue
// silently). The _WIN32 guard is a portability build guard matching the
// FilePtr::sync idiom above, not a runtime fallback: Windows uses _commit
// on the file, and directory fsync is not available via this interface.
static inline bool fsync_parent_dir(const char* path) {
#ifndef _WIN32
    std::string p(path);
    size_t slash = p.find_last_of('/');
    std::string dir = (slash == std::string::npos)
        ? std::string(".")
        : p.substr(0, slash == 0 ? 1 : slash);
    int dfd = open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (dfd < 0) {
        fprintf(stderr, "Error: could not open dir %s for fsync (errno=%d)\n", dir.c_str(), errno);
        return false;
    }
    bool ok = fsync(dfd) == 0;
    if (!ok) {
        fprintf(stderr, "Error: fsync of dir %s failed (errno=%d)\n", dir.c_str(), errno);
    }
    close(dfd);
    return ok;
#else
    (void) path;  // Windows: directory fsync not available via this path
    return true;
#endif
}

// Checked fwrite: verifies the full count was written. On failure prints an
// error and returns false; callers must propagate the failure (never produce
// corrupt files silently). Function form of the former CHECKED_WRITE macro.
static inline bool checked_write(const void* ptr, size_t size, size_t count, FILE* f) {
    if (fwrite(ptr, size, count, f) != count) {
        // No __FILE__:__LINE__ in this message: as a static inline it would
        // always report this header's location, not the caller's.
        fprintf(stderr, "Error: write failed (expected %zu items)\n", count);
        return false;
    }
    return true;
}

// FNV-1a 64-bit hash of a string: cheap content identity for the
// resume-time prompts-file check (v5 checkpoints store a rolling hash of
// every consumed line; v4 stored a single-line hash).
static inline uint64_t hs_fnv1a64(const std::string& s) {
    uint64_t h = 14695981039346656037ull;  // FNV-1a 64-bit offset basis
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;            // FNV prime
    }
    return h;
}

// Advance a running FNV-1a-64 state with one prompt line. The trailing
// newline is mixed in so the concatenation is unambiguous ("ab"+"c" vs
// "a"+"bc"). The checkpoint's content hash is this rolling state over all
// consumed lines: any change anywhere in the processed prefix changes it.
static inline uint64_t fnv_roll_line(uint64_t h, const std::string& line) {
    for (unsigned char c : line) {
        h ^= c;
        h *= 1099511628211ull;            // FNV prime
    }
    h ^= (unsigned char)'\n';
    h *= 1099511628211ull;
    return h;
}


// -- Batch-Accumulate Output Writer (io-util.cpp) ------------------------

// Write accumulated means to output.bin (binary accumulator format).
// temp+fsync+rename: a crash mid-write can never leave a truncated output.
bool write_batch_output(
    const AccumulatorMap& accumulators,
    const char* output_path,
    int32_t n_embd
);

// -- Checkpoint / Resume (io-util.cpp) ------------------------------------

// Run fingerprint: identifies the invocation that produced a checkpoint so
// --resume can refuse checkpoints written under different extraction
// settings (mode, generation length, token skip, target layers) or over a
// different prompts file. Resuming across differing settings would silently
// merge incompatible accumulators into one valid-looking output file.
struct checkpoint_fingerprint {
    bool    generate_mode   = false; // --generate N > 0 was active
    int32_t generate_tokens = 0;
    int32_t token_skip      = 0;
    std::vector<int32_t> layers;     // target layer indices; writer sorts, identity is order-independent
    // v4/v5 content identity: a hash over consumed prompts-file content and
    // the expected prompt count. v5 (current) stores a rolling FNV-1a-64
    // over every consumed line, newline-terminated: any change anywhere in
    // the processed prefix is caught. Zero fnv64 with progress > 0 is
    // rejected as a corrupt/hand-edited checkpoint. (v4 stored a hash of
    // only the last processed line on a retired basis; such checkpoints
    // resume with the count check only, plus a warning.)
    uint64_t content_fnv64  = 0;
    int32_t n_prompts       = 0;
};

// Write checkpoint: version + n_iterated + run fingerprint (v3) + content
// identity (v4) + accumulator state (binary accumulator format). The
// checkpoint file is output_path + ".checkpoint". fp.content_fnv64 and
// fp.n_prompts carry the v4 content identity.
bool write_checkpoint(
    const AccumulatorMap& accumulators,
    const char* output_path,
    int32_t n_embd,
    int32_t n_iterated,
    const checkpoint_fingerprint& fp
);

// Read checkpoint: restore accumulator state and return n_iterated (skip
// count). Returns false if checkpoint doesn't exist or is corrupt. For v3+
// checkpoints the stored run fingerprint must match `expected` exactly;
// v1/v2 checkpoints carry no fingerprint and are accepted with a warning
// (legacy files cannot be validated retroactively). v4 checkpoints also
// carry the content identity of the prompts file: when n_iterated > 0 and
// prompts_file is non-null, the hash of its n_iterated-th non-empty line and
// the stored prompt count must match expected (different content is a loud
// error). v3 files predate the content check and skip it with a warning.
bool read_checkpoint(
    const char* output_path,
    AccumulatorMap& accumulators,
    int32_t& n_iterated,
    int32_t expected_n_embd,
    const checkpoint_fingerprint& expected,
    const char* prompts_file
);
