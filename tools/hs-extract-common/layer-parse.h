// Shared layer-list parser for the hs-extract tools: 'all' or a
// comma-separated list of hidden_states indices (0 = embeddings, i = state
// entering block i (= HF hidden_states[i]), N = final block output; negative
// indices resolve Python-style from the end, -1 = last slot). Invalid tokens,
// out-of-range values (before resolution), and duplicates (after resolution)
// print a diagnostic to stderr and yield an empty vector; callers treat an
// empty result as a fatal argument error.
#pragma once

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// n_slots = n_layer + 1 (the hidden_states slot count).
inline static std::vector<int> hs_parse_layer_list(const char* str, int n_slots) {
    std::vector<int> out;
    if (std::string(str) == "all") {
        for (int i = 0; i < n_slots; i++) out.push_back(i);
        return out;
    }
    const char* p = str;
    while (*p) {
        char* endptr = nullptr;
        errno = 0;
        long val = strtol(p, &endptr, 10);
        if (endptr == p) {
            fprintf(stderr, "Error: invalid layer '%c' in layers string '%s'\n", *p, str);
            return {};
        }
        if (errno == ERANGE) {
            fprintf(stderr, "Error: layer value out of range in '%s'\n", str);
            return {};
        }
        // Out-of-range is decided before negative resolution: valid negatives
        // are exactly [-n_slots, -1].
        if (val >= n_slots || val < -n_slots) {
            fprintf(stderr, "Error: layer %ld out of range [0, %d) or [%d, -1]\n", val, n_slots, -n_slots);
            return {};
        }
        out.push_back((val < 0) ? (int)val + n_slots : (int)val);
        while (*endptr && *endptr != ',') endptr++;
        if (*endptr == ',') endptr++;
        p = endptr;
    }
    // Duplicate slots (including negative aliases such as "17,-1") would emit
    // the same layer twice; reject after resolution.
    std::vector<char> seen(n_slots > 0 ? n_slots : 0, 0);
    for (int l : out) {
        if (seen[l]) {
            fprintf(stderr, "Error: duplicate layer index %d in '%s' (after negative-index resolution)\n", l, str);
            return {};
        }
        seen[l] = 1;
    }
    return out;
}
