# H-Net from Scratch: Dynamic Chunking on Raw Bytes

Most language models read text through a tokenizer: a table of chunks built offline, before any learning starts. Hwang, Wang and Gu ("Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", 2025) remove it. An H-Net reads raw UTF-8 bytes, and a router trained together with the rest of the network decides which positions reach a large main network. This project builds that model from the paper's equations in PyTorch, on TinyStories and one CPU. You write bits per byte and the two baseline chunkers (fixed-width pooling and SpaceByte's space-like rule); the dynamic chunking pieces (the cosine router, batched compaction, the EMA over chunks, the causal upsampler with a straight-through confidence, the ratio loss); a small Mamba-2 layer with its chunked state space scan, a RoPE Transformer layer, a layout parser and the recursive stage that makes a complete byte-level model; the paper's stage learning rates (equation 11), a warmup-stable-decay schedule and its Appendix B FLOP formulas, which you check against its Table 2 (0.423 against 0.42 GFLOPs per byte for the Large Transformer); and a streaming step that keeps a KV cache, SSM states, one router key and one smoothed vector, and matches the full forward to about 1e-5 while showing which bytes woke the main network. The budget here is thousands of times smaller than the 30B to 200B bytes the paper needs before an H-Net passes a BPE Transformer, so the paper's headline result is not something this project can show, and there is no BPE Transformer arm. What you can measure, over seeds 0 to 2 at about 230 to 260 kFLOPs per byte, 1,200 steps (2.5 MB of text): isotropic byte model 1.587 bits per byte, space-like rule 1.697, two stages 1.717, dynamic chunking 1.732, dynamic without smoothing 1.753, fixed-width pooling 1.757. At 4,000 steps (8 MB): 1.451, 1.543, 1.574, 1.585 and pooling 1.627, with dynamic chunking compressing less there and costing 276 kFLOPs per byte. What reproduced: dynamic chunking beats pooling on every seed at both budgets; removing the EMA hurts on every seed (1,200 steps only, where it was run); the learning-rate multipliers help, 1.741 against 1.790 and 1.900 with flat rates at target 8; a target of 6 lands at 4.8 bytes per chunk at 1,200 steps, where the paper's does, then drifts to 4.2 by 4,000. A second stage helps on every seed at 1,200 steps and on two of three at 4,000. What did not reproduce: the paper's isotropic byte models come last, while here a Mamba-only model with under a third of the parameters beats every arm at both budgets; the paper's router ties the space-like rule, while here the rule wins on every seed; the paper's boundaries land mostly on whitespace, while here only 14 to 30 percent do and they catch 17 to 35 percent of the space-like positions, even at 4,000 steps; and removing cosine routing or the straight-through estimator does not make things worse at this budget. The compression swings the paper shows without smoothing appear only as a small increase in spread. The scaffold runs one seed at 500 steps in three to four minutes; a single run at that length can order the arms differently, which is why the sweep numbers are here.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_corpus
- [x] **2.** sample_batch
- [x] **3.** bits_per_byte
- [x] **4.** pool_boundaries
- [x] **5.** space_boundaries
- [x] **6.** RoutingModule
- [x] **7.** compact_chunks
- [x] **8.** smooth_chunks
- [x] **9.** causal_upsample
- [x] **10.** straight_through_confidence
- [x] **11.** ratio_loss
- [x] **12.** RMSNorm
- [x] **13.** ssd_scan
- [x] **14.** MambaLayer
- [x] **15.** apply_rope
- [x] **16.** TransformerLayer
- [x] **17.** build_stack
- [x] **18.** HNetStage
- [x] **19.** HNetLM
- [x] **20.** lr_multipliers
- [x] **21.** stage_param_groups

---

Built on Deep-ML.
