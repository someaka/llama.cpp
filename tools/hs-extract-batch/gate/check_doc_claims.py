#!/usr/bin/env python3
"""Shape check for the parity table in tools/server/README.md.

The table's numbers are measurements and are deliberately NOT pinned: they move
when the hardware, model or prompt changes. What is pinned is the cell
contract - every measured row must name its pooling mode, its layer set and
either `n` or a trial count (or state explicitly that n was not recorded), and
a row that quotes no number must say so. Rounds 8 and 9 of review each found a
cell that broke this contract while the audit stayed green; this check makes
that class mechanical.

Also pins three facts the docs derive from code, so a doc/code drift shows up
here rather than in review:
  * the hidden-state CLIs have no Flash-Attention flag,
  * the server prints 9 significant digits for float32,
  * `--kv-unified` is documented as the default when `-np` is unset.

Exit codes: 0 all checks pass, 1 at least one FAIL.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
SERVER_README = ROOT / "tools/server/README.md"
HS_EXTRACT = ROOT / "tools/hs-extract/hs-extract.cpp"
HS_BATCH = ROOT / "tools/hs-extract-batch/hs-extract-batch.cpp"
SERVER_HS = ROOT / "tools/server/server-hidden-states.cpp"

AXES = [
    "Flash-Attention",
    "Backend build",
    "Compute placement",
    "Concurrency, per-slot KV",
    "Concurrency, unified KV",
]

failures = []


def fail(msg):
    failures.append(msg)


def table_rows(text):
    """Return the data rows of the measured-deviations table."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("| Axis | Change from baseline | Metric, scope, value |"):
            start = i + 2  # skip the |---| separator
            break
    if start is None:
        fail("parity table header not found in tools/server/README.md")
        return []
    rows = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        rows.append(line)
    return rows


def main():
    if not SERVER_README.is_file():
        fail(f"missing {SERVER_README}")
        print(f"FAIL: {failures[0]}")
        sys.exit(1)
    rows = table_rows(SERVER_README.read_text())

    if len(rows) != len(AXES):
        fail(f"expected {len(AXES)} table rows, found {len(rows)}")

    seen = {}
    for row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) != 3:
            fail(f"row does not have 3 cells: {row[:80]}")
            continue
        axis = next((a for a in AXES if cells[0].startswith(a)), None)
        if axis is None:
            fail(f"unknown row: {cells[0][:60]}")
            continue
        _axis, _change, value = cells[0], cells[1], cells[2]
        seen[axis] = value

    for axis in AXES:
        if axis not in seen:
            fail(f"missing row: {axis}")
            continue
        value = seen[axis]
        if "No metric recorded" in value:
            # A qualitative row must say so, and must not smuggle a number in.
            if re.search(r"\d+(?:\.\d+)?e-?\d+|\d+\s*(?:x|%)", value):
                fail(f"{axis}: qualitative row quotes a numeric result")
            continue
        if "pool=" not in value:
            fail(f"{axis}: measured row does not name its pooling mode (pool=...)")
        else:
            # Every absolute figure must sit next to its pooling label; a row
            # that names pooling only somewhere else does not bind the number.
            for m in re.finditer(r"\babs\b(?=[^.()]{0,15}\d)", value):
                window = value[max(0, m.start() - 35):m.start() + 35]
                if "pool=" not in window:
                    fail(f"{axis}: an abs figure is not bound to a pooling mode within 35 chars")
                    break
        if "layer" not in value:
            fail(f"{axis}: measured row does not name its layer set")
        has_n = re.search(r"\bn=\d+", value) or re.search(r"\b\d+\s+trials\b", value)
        if not has_n and "n not recorded" not in value:
            fail(f"{axis}: measured row has neither n=, a trial count, nor an explicit 'n not recorded'")

    # Derived code facts.
    for cli, name in ((HS_EXTRACT, "hs-extract"), (HS_BATCH, "hs-extract-batch")):
        if cli.is_file():
            src = cli.read_text(errors="replace")
            if re.search(r'"-fa"|"--flash-attn"', src):
                fail(f"{name} has a Flash-Attention flag; the docs say the CLIs have none")

    if SERVER_HS.is_file():
        if '"%.9g"' not in SERVER_HS.read_text(errors="replace"):
            fail("server-hidden-states.cpp no longer prints with %.9g (docs promise 9 significant digits)")

    readme = SERVER_README.read_text()
    if "`--kv-unified`, the default when `-np` is unset" not in readme:
        fail("the unified-KV row no longer states the -np default")

    if failures:
        for msg in failures:
            print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"PASS: {len(rows)} parity rows honour the cell contract; 3 derived code facts hold")


if __name__ == "__main__":
    main()
