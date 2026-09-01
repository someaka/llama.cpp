#!/usr/bin/env python3
"""Generate a deterministic small assignments.bin + prompts.txt for the
run_batch golden gate. All prompts are >= 8 tokens (tokenized by the gate's
model, BOS added by binary), all explicit ranges stay within [0, 6) so every
range is valid for every prompt."""
import struct, sys, pathlib

# Default to the canonical input dir gate_batch.py reads (GOLD = /tmp/hs-batch-golden).
# The gate refuses to run if these fixtures are missing, and /tmp is wiped on
# reboot — so the default lands exactly where gate_batch.py expects them.
out_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("/tmp/hs-batch-golden/inputs")
out_dir.mkdir(parents=True, exist_ok=True)

# 24 prompts, varied lengths, ASCII only (deterministic), all well over 6 tokens.
prompts = [
    "The weather today is surprisingly calm and pleasant for the season.",
    "A short sentence about nothing at all.",
    "Researchers at the laboratory discovered a new method for measuring very small temperature changes in chemical reactions over long periods of time.",
    "Hello world, this is a test of the emergency broadcasting system.",
    "She walked into the room and immediately noticed the strange silence.",
    "Quantum computing promises dramatic speedups for certain classes of problems, but practical fault tolerance remains elusive.",
    "The cat sat on the mat and fell asleep in a patch of sunlight.",
    "Economic indicators suggest a moderate recovery, though unemployment remains stubbornly high in several regions across the continent.",
    "I am so happy to see you today!",
    "I am furious and outraged by this betrayal.",
    "The recipe calls for flour, sugar, butter, and two large eggs beaten until fluffy.",
    "Midnight came and went without a single telephone call.",
    "In the beginning the universe was extremely hot and dense.",
    "A gentle breeze carried the scent of jasmine through the open window of the old farmhouse kitchen.",
    "Debugging concurrent programs is one of the hardest problems in software engineering practice today.",
    "The train arrived exactly on time, which surprised everyone on the platform.",
    "Mountains rose in the distance, their peaks dusted with the first snow of autumn.",
    "He opened the letter with trembling hands and read the first line twice.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "The courtroom fell silent as the witness took the stand and was sworn in.",
    "Waves crashed against the seawall during the storm last night.",
    "Mathematics is the language with which God has written the universe, or so it has been said.",
    "The old library smelled of dust and paper and quiet afternoons.",
    "Everything changed the morning the power went out across the whole city.",
]

# 3 groups with names; 3 masks in use (0,1,2).
groups = ["neutral", "story", "probe"]

def assignment(gid, mid, mtype, skip=0, ranges=None):
    return (gid, mid, mtype, skip, ranges or [])

# Deterministic assignment mix per prompt: vary by index.
assigns_per_prompt = []
for i in range(len(prompts)):
    a = []
    # every prompt: one simple-skip assignment (skip 0,1,2 cycling)
    a.append(assignment(i % 3, 0, 0, skip=(i % 3)))
    # even prompts: explicit ranges (two ranges incl. adjacent pair)
    if i % 2 == 0:
        a.append(assignment((i + 1) % 3, 1, 1, ranges=[(1, 4), (2, 3)]))
    # every 3rd prompt: overlapping ranges on mask 2
    if i % 3 == 0:
        a.append(assignment((i + 2) % 3, 2, 1, ranges=[(0, 2), (1, 5)]))
    assigns_per_prompt.append(a)

# prompts.txt (non-empty lines only)
with open(out_dir / "prompts.txt", "w") as f:
    f.write("\n".join(prompts) + "\n")

# assignments.bin
with open(out_dir / "assignments.bin", "wb") as f:
    f.write(struct.pack("<i", 0x43524431))          # CRD1 magic
    f.write(struct.pack("<i", len(prompts)))         # n_prompts
    f.write(struct.pack("<i", 0))                    # n_embd (0 = no check)
    f.write(struct.pack("<i", len(groups)))          # n_groups
    for name in groups:
        b = name.encode()
        f.write(struct.pack("<i", len(b)))
        f.write(b)
    for a_list in assigns_per_prompt:
        f.write(struct.pack("<i", len(a_list)))
        for gid, mid, mtype, skip, ranges in a_list:
            f.write(struct.pack("<iii", gid, mid, mtype))
            if mtype == 0:
                f.write(struct.pack("<i", skip))
            else:
                f.write(struct.pack("<i", len(ranges)))
                for s, e in ranges:
                    f.write(struct.pack("<ii", s, e))

# Generation-mode assignments: --generate only accepts skip=0/mask_type=0.
with open(out_dir / "assignments_gen.bin", "wb") as f:
    f.write(struct.pack("<i", 0x43524431))
    f.write(struct.pack("<i", len(prompts)))
    f.write(struct.pack("<i", 0))
    f.write(struct.pack("<i", len(groups)))
    for name in groups:
        b = name.encode()
        f.write(struct.pack("<i", len(b)))
        f.write(b)
    for i in range(len(prompts)):
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<iiii", i % 3, 0, 0, 0))  # gid, mask 0, type 0, skip 0

print(f"wrote {out_dir/'prompts.txt'} ({len(prompts)} prompts), {out_dir/'assignments.bin'} and {out_dir/'assignments_gen.bin'}")
