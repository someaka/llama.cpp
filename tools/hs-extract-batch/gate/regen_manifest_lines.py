#!/usr/bin/env python3
"""Regenerate the adoption manifest's capture-site line column from HEAD.

The manifest's ADOPT table cites, per architecture, the line of the final
residual add (`cur = ggml_add(...)`) that feeds `capture_layer_output(il, cur)`.
That column rots as builders are edited; this script re-derives it from source.

Rule: find the capture call; walk back (max 60 lines) to the last line matching
`cur = ggml_add(`. Emit file:line pairs; compare with the manifest table.
"""
import os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MANIFEST = os.path.join(REPO, "docs/hidden-states-adoption-manifest.md")

def last_add_before(lines, capture_idx, window=60):
    for j in range(capture_idx - 1, max(0, capture_idx - window), -1):
        if re.search(r"\bcur\s*=\s*ggml_add\b", lines[j]):
            return j + 1  # 1-based
    return None

def build_truth():
    truth = {}
    models = os.path.join(REPO, "src/models")
    for fn in sorted(os.listdir(models)):
        if not fn.endswith(".cpp"):
            continue
        lines = open(os.path.join(models, fn)).read().splitlines()
        for i, line in enumerate(lines):
            if "capture_layer_output(" in line and "void" not in line:
                arch = fn[:-4]
                truth[arch] = last_add_before(lines, i)
                break  # first capture call in the file is the loop-bottom one
    return truth

def main():
    check_only = "--check" in sys.argv
    truth = build_truth()
    lines = open(MANIFEST).read().splitlines()
    pat = re.compile(r"^\| (\d+) \| (\S+) \| src/models/(\S+\.cpp) \| (\d+) \|$")
    fixed, mismatched, out = 0, [], []
    for line in lines:
        m = pat.match(line)
        if not m:
            out.append(line)
            continue
        _, arch, file_rel, old_line = m.groups()
        if arch not in truth or truth[arch] is None:
            out.append(line)
            continue
        new_line = truth[arch]
        if int(old_line) != new_line:
            mismatched.append((arch, int(old_line), new_line))
            fixed += 1
        out.append(f"| {m.group(1)} | {arch} | {file_rel} | {new_line} |")
    if check_only:
        if mismatched:
            print(f"FAIL: {len(mismatched)} manifest line(s) drifted from HEAD:")
            for arch, old_ln, new_ln in mismatched:
                print(f"  {arch}: manifest says {old_ln}, HEAD says {new_ln}")
            return 1
        print("PASS: manifest capture-site lines match HEAD")
        return 0
    open(MANIFEST, "w").write("\n".join(out) + "\n")
    print(f"rows corrected: {fixed}")
    for arch, old, new in mismatched:
        print(f"  {arch}: {old} -> {new}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
