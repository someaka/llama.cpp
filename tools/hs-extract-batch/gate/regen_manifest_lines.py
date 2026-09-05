#!/usr/bin/env python3
"""Regenerate/verify the adoption manifest's capture-site line column.

Rule: for every ADOPT-table row, open the row's file (authoritative join key),
find the LAST `capture_layer_output(` call in the file (helpers/MTP blocks may
define earlier calls), then walk back (max 60 lines) to the last line matching
`cur = ggml_add(` — that line number is the "final residual add" evidence.

--check mode verifies every row and FAILS if any row cannot be checked or
disagrees with HEAD.
"""
import os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MANIFEST = os.path.join(REPO, "docs/hidden-states-adoption-manifest.md")
ROW = re.compile(r"^\| (\d+) \| (\S+) \| (?:src/models/)?(\S+\.cpp) \| (\d+)([^|]*)\|$")

def capture_line_for(lines):
    """Index (0-based) of the LAST capture_layer_output call in the file."""
    last = None
    for i, line in enumerate(lines):
        if "capture_layer_output(" in line and "void" not in line \
           and "llm_graph_context" not in line:
            last = i
    return last

def add_line_before(lines, capture_idx, window=60):
    for j in range(capture_idx - 1, max(0, capture_idx - window), -1):
        if "ggml_add(" in lines[j]:
            return j + 1  # 1-based
    return None

def main():
    check_only = "--check" in sys.argv
    truth = {}
    models = os.path.join(REPO, "src", "models")
    for fn in sorted(os.listdir(models)):
        if fn.endswith(".cpp"):
            lines = open(os.path.join(models, fn)).read().splitlines()
            ci = capture_line_for(lines)
            if ci is not None:
                truth[fn[:-4]] = add_line_before(lines, ci)

    rows = open(MANIFEST).read().splitlines()
    out, mismatched, unchecked, fixed = [], [], [], 0
    total_adopt = 0
    for line in rows:
        m = ROW.match(line)
        if not m:
            out.append(line)
            continue
        total_adopt += 1
        num, arch, file_rel, old_ln, note = m.groups()
        stem = file_rel[:-4]  # basename without .cpp — authoritative join key
        if stem not in truth or truth[stem] is None:
            unchecked.append((arch, file_rel))
            out.append(line)
            continue
        new_ln = truth[stem]
        if int(old_ln) != new_ln:
            mismatched.append((arch, int(old_ln), new_ln))
            fixed += 1
            note = ""  # stale annotation described the old line's context
        out.append(f"| {num} | {arch} | {file_rel} | {new_ln}{note.rstrip()} |")

    if check_only:
        if unchecked:
            print(f"FAIL: {len(unchecked)} ADOPT rows could not be checked:")
            for arch, f in unchecked:
                print(f"  {arch} ({f})")
            return 1
        if mismatched:
            print(f"FAIL: {len(mismatched)} manifest line(s) drifted from HEAD:")
            for arch, old_ln, new_ln in mismatched:
                print(f"  {arch}: manifest says {old_ln}, HEAD says {new_ln}")
            return 1
        print(f"PASS: all {total_adopt} ADOPT rows match HEAD")
        return 0

    open(MANIFEST, "w").write("\n".join(out) + "\n")
    print(f"rows corrected: {fixed}; unchecked: {len(unchecked)}")
    for arch, old_ln, new_ln in mismatched:
        print(f"  {arch}: {old_ln} -> {new_ln}")
    for arch, f in unchecked:
        print(f"  UNCHECKED {arch} ({f})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
