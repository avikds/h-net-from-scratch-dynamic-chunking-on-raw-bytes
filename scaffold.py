"""
H-Net from Scratch: Dynamic Chunking on Raw Bytes scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""H-Net from Scratch: run the experiments at a small budget and read the results.

One seed of a short budget, on one CPU. Single runs at this length cannot settle small differences:
the project description reports the full sweep over three seeds at 1,200 and 4,000 steps."""
from solution import *
import time
import torch

torch.set_num_threads(1)
t_start = time.time()
STEPS = 500
data = load_corpus()
train_data, val_data = data[:15_000_000], data[15_000_000:]
print(f"{len(data):,} bytes of TinyStories; training on {STEPS * 16 * 128:,} bytes ({STEPS} steps of 16 x 128)")

print("\n1. Same compute per byte, different chunkers")
print(f"{'arm':14s}{'bits/byte':>10s}{'bytes/chunk':>13s}{'kFLOPs/byte':>13s}{'params':>9s}{'sec':>6s}")
runs = {}
for name in ("byte", "pool", "space", "dynamic", "no_smoothing"):
    runs[name] = r = run_arm(name, train_data, val_data, STEPS, seed=0, log_every=10)
    bpc = f"{r['bytes_per_chunk'][0]:.2f}" if r["bytes_per_chunk"] else "1.00"
    print(f"{name:14s}{r['bpb']:10.3f}{bpc:>13s}{r['kflops_per_byte']:13.1f}{r['params']:9d}{r['seconds']:6.0f}")

print("\n2. Compression in the second half of training (target 6)")
for name in ("dynamic", "no_smoothing"):
    s = compression_summary(runs[name]["trace"], 6.0)
    print(f"{name:14s} mean {s['mean']:.2f} bytes per chunk, spread {s['std']:.2f}, {100 * s['gap']:.0f}% of the target")

print("\n3. The streaming step reproduces the full forward")
model = runs["dynamic"]["model"]
g = torch.Generator().manual_seed(1)
x, _ = sample_batch(val_data, 1, 128, g)
logits, ran = stream_bytes(model, x)
model.eval()
with torch.no_grad():
    full, stats = model(x)
model.train()
print(f"largest logit difference {(logits - full).abs().max().item():.1e}; main network ran on {sum(ran)} of {len(ran)} bytes; same boundaries as the full forward: {ran == stats[0]['b'][0].long().tolist()}")

print("\n4. What the router learned")
text = "Once upon a time, there was a little girl named Lily. She loved to play outside in the sun."
for label, s in (("clean", text), ("spaces removed", text.replace(" ", ""))):
    ids = torch.tensor(list(s.encode("utf-8")))
    _, ran = stream_bytes(model, ids[None])
    b = torch.tensor(ran, dtype=torch.bool)
    print(f"{label} ({b.sum().item()} chunks in {len(ids)} bytes): {boundary_view(ids, b)}")
g = torch.Generator().manual_seed(5)
x, _ = sample_batch(val_data, 32, 128, g)
model.eval()
with torch.no_grad():
    _, stats = model(x)
model.train()
a = boundary_alignment(x, stats[0]["b"])
print(f"on held-out text: {100 * a['on_space']:.0f}% of boundaries are on a space-like boundary, {100 * a['after_space']:.0f}% are one byte after it, and they catch {100 * a['space_recall']:.0f}% of the space-like boundaries")

print(f"\nTotal {time.time() - t_start:.0f} s. Ranking at this length is not the result: see the project description for the seed sweep.")
print("The paper's claim is about 100B tokens; here the budget is about 1 MB.")

