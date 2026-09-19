// tools/server/server-hidden-states.cpp — fork-owned out-of-line serializers for
// server_task_result_hidden_states (moved verbatim from server-task.cpp; the
// struct declaration stays in server-task.h so the result variant chain is
// untouched). Field parity between the two forms is pinned by fork-ci's
// envelope-parity check, which greps this file.
#include "server-hidden-states.h"

#include <cstdio>

//
// server_task_result_hidden_states
//
json server_task_result_hidden_states::to_json() {
    // Generic form (used by task-pipeline consumers): doubles as-is. The
    // /hidden-states endpoint writes to_json_text() instead -- nlohmann's
    // double writer re-expands ~0.3% of %.9g values past 9 digits, so the
    // wire form is built as text there.
    // Memory note (HS-4): each layer vector is deep-copied and widened to a
    // nlohmann double array — roughly 2-6x the raw payload (plus node
    // overhead) on top of the captured floats. At the 25M-float pool=none
    // cap that is a transient ~200-400 MB spike; prefer to_json_text() on
    // any path that handles cap-sized payloads.
    json layers = json::object();
    for (const auto & kv : hidden_states) {
        layers[std::to_string(kv.first)] = kv.second;
    }
    return json {
        {"index",            index},
        {"hidden_states",    layers},
        {"tokens_evaluated", n_tokens},
    };
}

std::string server_task_result_hidden_states::to_json_text() const {
    // Same envelope as to_json(), but every value is a %.9g literal -- the
    // CLI tools' exact text precision, emitted as raw JSON numbers (9
    // significant digits round-trip float32 losslessly).
    // Field parity is pinned by fork-ci's envelope-parity check: adding a
    // field to to_json() without mirroring it here fails CI, so the two
    // implementations cannot silently drift.
    auto render_values = [](const std::vector<float> & vec) {
        std::string out = "[";
        for (size_t i = 0; i < vec.size(); i++) {
            if (i > 0) {
                out += ",";
            }
            char buf[32];  // worst-case %.9g is 15 chars; 32 gives headroom
            snprintf(buf, sizeof(buf), "%.9g", (double) vec[i]);
            out += buf;
        }
        out += "]";
        return out;
    };
    // Envelope matches to_json()'s shape (one result object per call).
    std::string out = "{\"index\":" + std::to_string(index) + ",\"hidden_states\":{";
    bool first_layer = true;
    for (const auto & kv : hidden_states) {
        if (!first_layer) {
            out += ",";
        }
        first_layer = false;
        out += "\"" + std::to_string(kv.first) + "\":" + render_values(kv.second);
    }
    out += "},\"tokens_evaluated\":" + std::to_string(n_tokens) + "}";
    return out;
}