// Internal shared header for hs-extract-batch: RAII FILE wrapper, checked
// writes, and durable/atomic I/O helpers (checkpoint read/write, atomic
// temp+fsync+rename finalization). Extracted verbatim from
// hs-extract-batch.cpp (pure code motion).
#pragma once

#include <cstdio>
#include <cstdint>
#ifndef _WIN32
#include <unistd.h>  // fsync for durability before rename (POSIX)
#endif
#include <string>

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

// Wraps fwrite to check return value. On failure, prints an error and
// returns false from the calling function. Used for all output writes
// to ensure corrupt files are never silently produced.
#define CHECKED_WRITE(ptr, size, count, f)                                      \
    do {                                                                        \
        if (fwrite((ptr), (size), (count), (f)) != (size_t)(count)) {          \
            fprintf(stderr, "Error: write failed at %s:%d (expected %zu, wrote less)\n", \
                    __FILE__, __LINE__, (size_t)(count));                      \
            return false;                                                       \
        }                                                                       \
    } while (0)

// -- Batch-Accumulate Output Writer (io-util.cpp) ------------------------

// Write accumulated means to output.bin (binary accumulator format).
// temp+fsync+rename: a crash mid-write can never leave a truncated output.
bool write_batch_output(
    const AccumulatorMap& accumulators,
    const char* output_path,
    int32_t n_embd
);

// -- Checkpoint / Resume (io-util.cpp) ------------------------------------

// Write checkpoint: version + n_iterated + accumulator state (binary
// accumulator format). The checkpoint file is output_path + ".checkpoint".
bool write_checkpoint(
    const AccumulatorMap& accumulators,
    const char* output_path,
    int32_t n_embd,
    int32_t n_iterated
);

// Read checkpoint: restore accumulator state and return n_iterated (skip
// count). Returns false if checkpoint doesn't exist or is corrupt.
bool read_checkpoint(
    const char* output_path,
    AccumulatorMap& accumulators,
    int32_t& n_iterated,
    int32_t expected_n_embd
);
