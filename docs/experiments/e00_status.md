# E00 baseline infrastructure

## Verified on Apple Silicon

The repository was initialized on `main`. Three upstream gitlinks and
the two Hugging Face checkpoint revisions, filenames, and published SHA-256
hashes are pinned. An initial sandbox blocked the Git write and GitHub DNS
resolution. Both operations succeeded after the workspace permissions changed.
The Mac hash lock installed under Python 3.12.13, Torch 2.8.0, and torchvision
0.23.0. `uv pip check` passed. The Linux x86-64 CUDA 12.8 lock resolved and
passed a cross-platform hash-lock sync dry run from Mac.

```sh
scripts/bootstrap.sh
scripts/doctor.sh
scripts/run.sh prepare --config configs/experiment/e00_synthetic.yaml
scripts/run.sh evaluate --config configs/experiment/e00_synthetic.yaml
scripts/run.sh evaluate --config configs/experiment/e03_correspondence.yaml
.venv/bin/pytest -q
.venv/bin/ruff check src tests
uv pip sync --require-hashes --python-platform x86_64-manylinux_2_28 --dry-run env/linux-cu128.lock.txt
```

The CPU suite passed 15 tests and skipped two opt-in pretrained tests.
E00 fixture evaluation yielded zero depth RMSE and zero Chamfer-L1 for both
point outputs with 4,121 visible points across three views. E03 fixture
evaluation produced six pair-by-site rows, two patch sizes per pair, and a
recall-by-angle plot. The fixture deliberately uses known geometry to verify
the metric implementation. These observations are pipeline checks.

## Pinned model contract

- VGGT construction counted **1,256,537,516** parameters. Its 1,797 state tensor
  names and shapes matched the published safetensors header. A random-weight
  two-view CPU forward at 28×42 returned pose `[1,2,9]`, depth `[1,2,28,42,1]`,
  and pointmap `[1,2,28,42,3]`. Local depth unprojection matched the upstream
  helper with maximum absolute difference `2.98e-8` in that check.
- VGGT patch embedding exposes DINO's `x_norm_patchtokens` before frame/global
  attention. The pinned aggregator retains zero-based slots 4, 11, 17 and 23,
  with frame features concatenated before global features. Its returned
  `patch_start_idx` was 5. The random-weight last spatial output was
  `[1,2,2,3,2048]`. The adapter uses the official quick-start crop loader
  and records its resize, center crop, and mixed-aspect padding.
- Sapiens 2 construction counted **113,993,472** parameters. Its 210 state
  tensor names and shapes matched the published safetensors header. A
  random-weight CPU forward on native 1024×768 input returned
  `[1,3081,768]` and the model reported nine prefix tokens. The spatial
  tensor therefore has a 64×48 patch grid. Extract selected blocks with
  `out_indices`. The last block uses the upstream final RMSNorm.
  Preserve the pinned processor's bilinear resize and ImageNet mean/std.

The model counts came from constructing the pinned source on Mac. The
checkpoint comparison read safetensors headers with HTTP byte-range requests.
Full weight bytes and SHA-256 validation are part of the pretrained run.

## Decisions and outstanding evidence

- Use FP32 for initial parity and a two-view synthetic input to keep the CUDA
  run bounded. Use one sequence-wide Sim(3) from non-collinear camera centers
  for metric reconstruction when those anchors are available. E00 evaluates
  the designated GT foreground and reports confidence arrays without thresholding. Degenerate
  sequences keep their raw exports and explicitly report unavailable alignment.
- E03 currently processes one synchronized tuple at a time. For each pair, the
  probe selects common GT-visible candidates within all compared feature grids
  and deduplicates rounded target pixels. PCK is in original image pixels and
  surface errors are normalized by the visible cloud bounding-box diagonal.
  Identity-level intervals, reference-order ablations, and semantic hard
  negatives require a verified dataset cohort.
- Pretrained VGGT parity and the frozen pretrained Sapiens smoke still need
  an authorized GPU host with the pinned checkpoints. Both real-weight
  integration tests are opt-in. THuman2.0 rendering is pending dataset
  access. See the [rendering plan](../decisions/thuman_rendering.md).
- Transfer work is limited to a linear projector building block. Training,
  scientific correspondence results, and complete-scan metrics remain
  roadmap milestones.

The next runnable experiment is the bounded VGGT parity check on Linux.
Checkpoint acquisition and environment setup are in the [remote instructions](../../README.md#linux-nvidia).

```sh
scripts/doctor.sh
scripts/run.sh evaluate --config configs/experiment/e00_vggt.yaml
```
