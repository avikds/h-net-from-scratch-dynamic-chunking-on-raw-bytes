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
- [x] **22.** wsd_factor
- [x] **23.** layer_flops
- [x] **24.** flops_per_byte
- [x] **25.** evaluate_bpb
- [x] **26.** train_hnet
- [x] **27.** IndependentRouter
- [x] **28.** arm_config
- [x] **29.** run_arm
- [x] **30.** compression_summary
- [x] **31.** mamba_step
- [x] **32.** transformer_step
- [x] **33.** stack_step
- [x] **34.** stage_step
- [x] **35.** stream_bytes
- [x] **36.** boundary_view
- [x] **37.** boundary_alignment

## Results

```
19,447,282 bytes of TinyStories; training on 1,024,000 bytes (500 steps of 16 x 128)

1. Same compute per byte, different chunkers
arm            bits/byte  bytes/chunk  kFLOPs/byte   params   sec
byte               1.758         1.00        231.8   106844   111
pool               1.981         4.92        250.5   383584    77
space              1.875         4.99        248.4   383584    76
dynamic            1.939         5.01        248.1   383584    87
no_smoothing       1.976         5.08        246.1   383584    86

2. Compression in the second half of training (target 6)
dynamic        mean 4.79 bytes per chunk, spread 0.32, 80% of the target
no_smoothing   mean 4.91 bytes per chunk, spread 0.54, 82% of the target

3. The streaming step reproduces the full forward
largest logit difference 1.3e-05; main network ran on 19 of 128 bytes; same boundaries as the full forward: True

4. What the router learned
clean (26 chunks in 91 bytes): Once u|pon a| tim|e|,| |t|h|ere was a| |littl|e gi|rl name|d Lily|. |Sh|e lo|v|ed to play |ou|t|sid|e in t|h|e sun|.
spaces removed (20 chunks in 73 bytes): Onceuponatime|,|th|erewasalittl|e|girlname|d|Lily|.S|h|e|lo|v|edto|playout|sid|ei|nth|esun|.
on held-out text: 19% of boundaries are on a space-like boundary, 5% are one byte after it, and they catch 18% of the space-like boundaries

Total 464 s. Ranking at this length is not the result: see the project description for the seed sweep.
The paper's claim is about 100B tokens; here the budget is about 1 MB.
```
