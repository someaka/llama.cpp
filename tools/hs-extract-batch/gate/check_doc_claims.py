#!/usr/bin/env python3
"""Shape check for the parity table in tools/server/README.md.

The table's numbers are measurements and are deliberately NOT pinned: they move
when the hardware, model or prompt changes. What is pinned is the cell
contract - every measured row must name a defined metric (abs, rel-L2,
rel-max), its pooling mode, a concrete layer set and either `n` or a trial
count (or state explicitly that n was not recorded). A row that quotes no
number must say "No metric recorded" and quote no measurement-looking number
(ISO dates and model names such as gemma-4-E2B are not measurements).

The check is vocabulary-driven rather than spelling-specific: a row must name
one of the protocol's metrics (abs, rel-L2, rel-max, bit-identical), and any
figure that carries one of those names must sit next to its pooling label. It
cannot tell an invented metric name from ordinary prose - a writer who renames
a metric and keeps every other cell honest is a review question, not a shape
question - so the vocabulary test is a presence test, not a proof.

Also pins three facts the docs derive from source, and missing sources are
failures (never skips):
  * the hidden-state CLIs have no Flash-Attention flag,
  * the server prints 9 significant digits for float32,
  * `-np` unset (auto) turns the unified KV pool on.

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
SERVER_MAIN = ROOT / "tools/server/server.cpp"

AXES = [
    "Flash-Attention",
    "Backend build",
    "Compute device",
    "Host op placement",
    "Concurrency, per-slot KV",
    "Concurrency, unified KV",
]

# Metric names must come from the protocol's own vocabulary.
METRIC_NAMES = r"\b(?:abs|rel-L2|rel-max|bit-identical)\b"
METRIC_WITH_NUMBER = METRIC_NAMES + r"(?=[^.()]{0,15}\d)"
LAYER_SET = r"layers?\s*\[?\s*\d"
ISO_DATE = r"\b20\d\d-\d\d-\d\d\b"
# Numbers that look like a measurement (model names such as gemma-4-E2B do not).
RESULT_NUMBER = r"\d+\.\d+|\b\d+\s*e-?\d+|\b\d+/\d+\b|\b\d+\s*(?:x|%|elements)\b"

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


def check_row(axis, value):
    qualitative = value.startswith("No metric recorded")
    if "No metric recorded" in value and not qualitative:
        fail(f"{axis}: 'No metric recorded' must open a qualitative row, not be a phrase inside a measured one")
    if qualitative:
        # Qualitative row: no measurement-looking numbers (an ISO date and a
        # model name are not measurements).
        if re.search(RESULT_NUMBER, re.sub(ISO_DATE, "", value)):
            fail(f"{axis}: qualitative row quotes a numeric result")
        return

    if not re.search(METRIC_NAMES, value):
        fail(f"{axis}: measured row names none of the defined metrics (abs, rel-L2, rel-max)")

    if "pool=" not in value:
        fail(f"{axis}: measured row does not name its pooling mode (pool=...)")
    else:
        # Every metric figure must sit next to its pooling label; naming
        # pooling somewhere else in the row does not bind the number.
        for m in re.finditer(METRIC_WITH_NUMBER, value):
            window = value[max(0, m.start() - 35):m.start() + 35]
            if "pool=" not in window:
                fail(f"{axis}: a '{m.group(0)}' figure is not bound to a pooling mode within 35 chars")
                break

    if not re.search(LAYER_SET, value):
        fail(f"{axis}: measured row does not name a concrete layer set (e.g. 'layers 4-5', 'layer 64')")
    elif "hs-fixture" in value and "qwen35" in value:
        # Two models in one row: each arm needs its own layer set, otherwise
        # one arm's numbers cannot be reproduced.
        if len(re.findall(LAYER_SET, value)) < 2:
            fail(f"{axis}: row measures two models but names a layer set for only one arm")

    has_n = re.search(r"\bn=\d+", value) or re.search(r"\b\d+\s+trials\b", value)
    if not has_n and "n not recorded" not in value:
        fail(f"{axis}: measured row has neither n=, a trial count, nor an explicit 'n not recorded'")


def main():
    if not SERVER_README.is_file():
        print(f"FAIL: missing {SERVER_README}")
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
        if axis in seen:
            fail(f"duplicate row: {axis}")
            continue
        seen[axis] = cells[2]

    for axis in AXES:
        if axis not in seen:
            fail(f"missing row: {axis}")
            continue
        check_row(axis, seen[axis])

    # Code facts. Missing sources fail closed.
    for cli, name in ((HS_EXTRACT, "hs-extract"), (HS_BATCH, "hs-extract-batch")):
        if not cli.is_file():
            fail(f"missing {cli.name}; cannot verify the {name} FA-flag fact")
            continue
        if re.search(r'"-fa"|"--flash-attn"', cli.read_text(errors="replace")):
            fail(f"{name} has a Flash-Attention flag; the docs say the CLIs have none")

    if not SERVER_HS.is_file():
        fail("missing server-hidden-states.cpp; cannot verify the %.9g fact")
    elif '"%.9g"' not in SERVER_HS.read_text(errors="replace"):
        fail("server-hidden-states.cpp no longer prints with %.9g (docs promise 9 significant digits)")

    if not SERVER_MAIN.is_file():
        fail("missing server.cpp; cannot verify the kv-unified default")
    else:
        src = SERVER_MAIN.read_text(errors="replace")
        if "n_parallel is set to auto" not in src or "params.kv_unified = true" not in src:
            fail("server.cpp no longer pairs the n_parallel=auto branch with kv_unified = true "
                 "(the table says --kv-unified is the default when -np is unset)")

    if failures:
        for msg in failures:
            print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"PASS: {len(rows)} parity rows honour the cell contract; 3 code facts hold")


if __name__ == "__main__":
    main()
