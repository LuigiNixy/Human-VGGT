# Human-centric feature alignment for VGGT
Implementation brief for Codex · 2026-11-18

## 1. Research objective
Investigate whether a frozen human-centric foundation model (Sapiens 2) can transfer useful priors into a generic geometry foundation model (VGGT), improving free-form 3D human reconstruction without a parametric body model, a human-specific reconstruction head, or a teacher at inference. Preserve general scene reconstruction. A failed hypothesis is a valid result. Do not claim novelty or superiority before evaluation.

Separate three questions: (H1) are teacher representations more useful for human surface correspondence? (H2) can VGGT representations absorb them? (H3) does this improve geometry on unseen people and real captures without degrading generic geometry? CKA and segmentation probes are diagnostics, not proof of H3. Single-view outputs represent visible surfaces, not automatically complete watertight meshes; multi-view is the primary task.

## 2. Upstream repositories and checkpoints
Initialize a new private parent repository, not a fork of an upstream model. From its root:

```bash
git init -b main
mkdir -p third_party
git submodule add https://github.com/facebookresearch/vggt.git third_party/vggt
git submodule add https://github.com/facebookresearch/sapiens2.git third_party/sapiens2
git submodule add https://github.com/facebookresearch/dinov2.git third_party/dinov2
git submodule status --recursive
```

Original VGGT is the primary student (facebook/VGGT-1B research checkpoint); its official repo includes training code. Sapiens 2 pretrained encoder is the primary teacher: start with facebook/sapiens2-pretrain-0.1b, then scale to 0.4B, 1B and 5B only after accounting for memory and resolution. Sapiens 2 standalone backbone needs only Torch and safetensors; use it instead of installing the full task stack initially. Official native resolution is H=1024, W=768, patch 16. The DINOv2 control should use the same VGGT patch encoder checkpoint/config where possible; standalone DINOv2 Base/Large are additional controls, not interchangeable with VGGT's own embedding.

Optional, only once primary baseline works:
- https://github.com/facebookresearch/vggt-omega (facebook/VGGT-Omega): newer generic geometry baseline and transfer replication; separate adapter/config, do not assume API compatibility.
- https://github.com/facebookresearch/sam-3d-body : single-view parametric comparator, gated checkpoint; MHR output, not a constraint for our method. Separate environment.
- https://github.com/YuliangXiu/ICON : reference scan-rendering/evaluation protocol; clone only if needed, do not import its SMPL dependencies into the main pipeline.

Read upstream licenses and model cards before use. Research VGGT weights and commercial weights have different licenses; THuman is noncommercial research data. Never commit weights, licensed datasets, credentials, or derived restricted assets. Pin actual submodule SHAs and HF revisions/checksums in manifests, not invented hashes. Use upstream docs to resolve exact checkpoint filenames.

## 3. Environment and deployment
Local Apple Silicon Mac (32 GB unified memory): write code, run offline CPU unit tests and small optional MPS smoke tests. GPU-specific operations may not support MPS; never require local full-model runs. Remote Linux NVIDIA GPU is authoritative for training. Start with Python 3.12, Torch >=2.7 (Sapiens 2 full repo requirement), matching torchvision, numpy, scipy, Pillow, safetensors, huggingface_hub, PyYAML, pytest, ruff, tqdm, matplotlib, trimesh, einops; use standard Torch SDPA where supported. These are proposed starting dependencies, not a tested lock. Inspect upstream pyproject/requirements and select a compatible exact pair; do not blindly install incompatible upstream pins over working Torch. Separate platform locks and install optional extras only when needed.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Mac only: select compatible CPU/MPS wheels
python -m pip install torch torchvision
python -m pip install numpy scipy pillow safetensors huggingface_hub pyyaml pytest ruff tqdm matplotlib trimesh einops
# Linux: install matching torch/torchvision from https://pytorch.org/get-started/locally/
# based on driver and CUDA wheel support; then install same common deps
python -m pip check
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available())'
```

Agent must produce pyproject.toml, env/common.in, env/mac.lock.txt, env/linux-cu*.lock.txt after actual resolution; record GPU name, driver, torch/CUDA versions and free memory. Do not modify system Python or drivers, sudo, or create paid cloud resources without permission. A100/H100 40–80 GB is a useful starting remote target, not a verified requirement. Start B=1, S=2, AMP bf16 only if supported; fp32 metrics and sensitive losses; test gradient checkpointing and accumulation before scaling. Cache frozen teacher features on server with dataset/model/transform/revision hashes; store fp16 or bf16 with validation against fp32. Use chunking and do not keep every intermediate by default.

Remote workflow: after committing locally, push to a user-authorized private remote, then clone/pull on server and initialize submodules. No server hostname, credentials or scheduler assumed. For unpushed development use rsync dry-run then rsync (no --delete), excluding .git, .venv, data, checkpoints, caches, outputs. Use environment variables HUMAN_VGGT_DATA, HUMAN_VGGT_WEIGHTS, HUMAN_VGGT_RUNS for server storage; keep local overrides ignored. Supply scripts/bootstrap.sh, scripts/doctor.sh, scripts/sync.sh and scripts/run.sh; optional Slurm template with account/partition/GPU settings configurable. Never silently start an expensive training run; execute bounded smoke tests, then provide command for larger runs if remote access/budget is unavailable.

## 4. Repository structure
```
human-vggt/
  AGENTS.md  RESEARCH_PLAN.md  README.md  pyproject.toml  .gitignore
  .agents/skills/{research-execution,experiment-integrity}/SKILL.md
  third_party/{vggt,sapiens2,dinov2}/
  env/                         # dependency inputs and platform locks
  configs/{model,data,experiment,local,remote}/
  src/human_vggt/
    models/{vggt_adapter,sapiens_adapter,projector}.py
    data/{schema,transforms,synthetic,thuman,real_capture}.py
    geometry/{camera,projection,alignment,visibility}.py
    losses/{feature,geometry}.py
    evaluation/{correspondence,reconstruction,metrics}.py
    cli.py  train.py  provenance.py
  scripts/                      # bootstrap, doctor, sync, render, run
  tests/{unit,integration}/
  manifests/{upstream,datasets,splits}/
  docs/{decisions,experiments,results}/
  data/  checkpoints/  cache/  outputs/  # ignored; external storage preferred
```
Use src layout and python -m human_vggt.cli; install own package editable. Keep adapters narrow, no copy-paste upstream architectures. Shared geometry and preprocessing must be used consistently by all baselines. Use upstream patches only if necessary; document/commit them in parent as explicit patches or a separate fork.

## 5. Benchmarks and data acquisition
A. Synthetic fixture (immediate, no external access): render an asymmetric textured primitive or toy articulated shape at 2–4 known cameras; deterministic RGB, masks, optical-axis depth, surface IDs and 3D points. Use to verify camera conventions, reprojection and feature sampling; not human evidence.

B. THuman2.0 (primary controlled human benchmark): https://github.com/ytrock/THuman2.0-Dataset . Request official access; original release contains 500 high-quality human scans. Render original clothed textured scan meshes, NOT SMPL fits. Use a headless Blender or equivalent reproducible renderer, with cameras at multiple elevations and azimuths, record intrinsics/extrinsics and exact depth/visibility. Begin 10 authorized scans for pipeline debugging; propose 400/50/50 train/val/test only after confirming subject IDs and usable count; if identities repeat, group by identity, use actual group counts instead. No frame/view leakage. For publication comparison use separately named official/prior-work splits when compatible. Start 16 views per scan, sample 1/2/4/8 views using fixed tuples; hold out backgrounds, texture/lighting conditions and viewpoints as stress tests. Keep render manifests and checksums. Do not treat renderer test as real capture.

C. Real multi-view: PKU-DyMVHumans https://pku-dymvhumans.github.io/ (geometry/reconstruction benchmark candidate); investigate release, licensing, calibrated synchronized images, scan geometry and confidence. DNA-Rendering https://dna-rendering.github.io/ (calibrated human-centric views/masks/SMPL-X assets; useful for real correspondence/reprojection, not automatically scan ground truth). Hi4D https://yifeiyin04.github.io/Hi4D/ (textured scans and close interactions; harder held-out extension). Select one with accessible independent geometry GT; avoid claiming mesh accuracy against fitted SMPL as scan GT. If unavailable, report real image diagnostics separately; do not block synthetic/THuman work. Synchronized views only for static VGGT, not different times in dynamic scenes.

D. Generic regression: ETH3D https://www.eth3d.net/overview (calibrated generic scenes and scan-based MVS benchmark). Start small training-scene subset with available GT, then full reproducible protocol. Optional Co3D camera-pose evaluation branch in upstream VGGT, with official splits. Keep scene-level independent test set; no human fine-tuning on evaluation scenes.

Dataset schema: sample_id, subject_id, scene_id, split, frame/time_id, view_id, rgb_path, mask_path, depth_path, K (3x3), T_cw (4x4 world-to-camera), width,height, depth_unit, depth_type, mesh_path, valid_mask, provenance. Missing fields explicitly null with capabilities flags. Standardize meters, RGB, OpenCV camera (+x right,+y down,+z forward), pixel-center convention documented. Renderer depth may be ray distance rather than z; convert explicitly. Test projection/unprojection and camera axes before trusting data. No GT camera or mask as student input in primary RGB-only evaluation; masks allowed in loss/evaluation, oracle-masked input is separate ablation.

## 6. Experiment ladder
E00 / baseline infrastructure: dataset fixture, camera tests, model adapters, inference and evaluation. Match official VGGT outputs on same images, preprocessing, eval mode, weights, device and tolerance; export camera, direct pointmap and depth-unprojected points separately. Sapiens output must be traced to original source. Record exact model parameter counts and shapes. Exit: tests pass, manifest produced and baseline reproduced or concrete remote command supplied.

E01 / representation extraction: sample the same physical RGB coordinates under independently transformed views. VGGT: distinguish patch embedding before aggregation from frame/global attention features. Official aggregator returns aggregated tokens and patch_start_idx; remove nonspatial camera/register tokens with actual index, not guessed constants. Trace output slots and layer indices in pinned repo, including frame/global concatenation. Sapiens 2: standalone encoder outputs dense features; expose selected layers with verified hooks. Never assume layer count or tensor layout. Standardize output [B,S,h,w,C] plus pixel transform and layer metadata. Sample normalized depths 25/50/75/100% where available; use last layer first. Use original image coordinate mappings, separate normalization, resize/pad (or crop with explicit validity) and effective intrinsics; never interpolate raw flattened tokens. Avoid forcing identical resolutions; compare native resolution and matched-resolution controls (only supported sizes). Report patch sizes, effective input area and pixel coverage.

E02 / diagnostic probes: spatially matched token samples, centered linear CKA (FP64 or stable FP32, subset fixed points); train linear/ridge mapping on training subjects, evaluate held-out subjects with cosine and normalized MSE. PCA fit on train only; visualize same basis in test. Include matched DINO/DINO control, random teacher, shuffled correspondence control, and teacher scale/resolution controls. Optional part probe with independently annotated labels: left/right limbs and anterior/posterior trunk only if those labels truly exist. Metrics: balanced accuracy/mIoU, a confusion matrix, subject split. Do not infer front/back labels from coarse trunk annotations. No claim that linear alignment must be possible for nonlinear transfer to work.

E03 / multi-view surface correspondence (priority): for surface point X visible in synchronized views i,j, project to both using GT K,T; reject out-of-frame, background, occluded or depth-inconsistent points. Use z-buffer or mesh visibility, not student depth. Sample at feature-grid coordinates using stored transforms and bilinear sampling. Compare within-image patch features, Sapiens independently encoded per view, VGGT patch embedding, VGGT aggregated features with identical view tuples. Report matched cosine vs negatives, retrieval recall@1/5, pixel PCK at stated thresholds, normalized surface distance, stratified by baseline angle and body region when valid. Hard negatives include nearby surface, symmetric limbs and same texture. Ensure no train/test identity overlap. Repeat with changed order/reference camera to test context dependence; bootstrapped confidence intervals by subject/scene. High cosine alone can be collapse; report negative separation and effective feature rank.

E04 / feature transfer: frozen Sapiens teacher eval/no gradient. First train a linear student->teacher projector with frozen VGGT as a diagnostic ONLY (cannot improve geometry). Then update last few student patch-encoder blocks or configurable LoRA (separate alternatives) while freezing aggregator and existing geometry heads initially. Compare alignment at patch embedding vs selected aggregation layers, one factor at a time. Later allow limited geometry transformer tuning only if needed. Student inference takes RGB only, through original geometry heads; remove projector and teacher. Unit-test identity of disabled branch.

For aligned foreground samples p: L_feat = mean[1 - cos(normalize(P(F_v(p))), normalize(stopgrad(F_s(p))))]. L_keep = normalized feature or output distillation from frozen original VGGT on generic and human input (same coordinates); use valid masks and separate coefficients. L_geom = existing upstream geometry supervision (depth, pointmap, camera; inspect official train configuration and correct scale/frame before using), applied only where real GT available. Total = lambda_feat L_feat + lambda_keep L_keep + lambda_geom L_geom. Start with normalized component logging; sweep lambda_feat {0.1,1} vs fixed geometry/keep weights, then adjust based on gradients/validation; not magic constants. Freeze schedule and optimizer state recorded. AdamW initial LR 1e-5 student / 1e-4 projector, wd 0.01, clip norm 1, 500-step pilot (adjust to actual data/memory), 3 seeds for confirmatory comparison. Teacher never optimized. Generic replay optional but test it as own variable. Log gradient norms by block, teacher gradient absent, nonzero student gradients, memory and loss components. No-loss baseline, geometry-only fine-tune with matched trainable parameters, feature-only, feature+geometry, random/shuffled or DINO teacher controls, and retention loss ablation. Match data, optimizer budget, seed, views and resolution whenever possible.

E05 / geometry evaluation: original VGGT vs all variants on held-out THuman, real capture (when GT available), generic ETH3D. Compare direct world pointmap and depth+camera unprojection as separate outputs. Since VGGT geometry may be gauge/scale ambiguous, report explicit alignment protocol: raw metric only if metric scale justified; one sequence-wide Sim(3) estimated via camera centers when nondegenerate or independently defined correspondences; single-view/degenerate camera fallback to documented point alignment (oracle result, separate). Never per-view align for multi-view consistency. No ICP as primary result; optional ICP upper bound separately. Fit on allowed alignment data, apply transform to all views. Report visible-surface and complete-scan coverage separately. Use identical sampling/density/confidence thresholds for methods, report sensitivity curves; do not use ground-truth visibility to silently select student predictions outside designated evaluation mask.

Metrics: accuracy (pred->GT nearest surface distance), completeness (GT->pred), symmetric Chamfer-L1 defined as half the sum of mean directional Euclidean distances, F-score at 1/2/5 cm in metric-aligned coordinates, depth AbsRel and RMSE with stated scale protocol, camera relative rotation/translation direction error where applicable, surface normal angular error when estimated comparably, cross-view point consistency, retained non-human generic performance. Report medians and mean across subjects plus subject bootstrap intervals, failures and coverage; separate native vs oracle-aligned and full vs visible. Confidence thresholds chosen on val, not test. For single-view SAM 3D Body, evaluate only comparable visible geometry under same masks/units; full mesh metrics separately note its different output prior. No forced mesh fitting/postprocessing for primary method.

## 7. Initial config contract (example, not implemented yet)
```yaml
experiment: e03_correspondence
seed: 42
device: auto              # CPU locally; explicit cuda remotely
precision: fp32
paths:
  data: ${HUMAN_VGGT_DATA}
  weights: ${HUMAN_VGGT_WEIGHTS}
  runs: ${HUMAN_VGGT_RUNS}
student:
  family: vggt
  checkpoint: facebook/VGGT-1B
  feature_sites: [patch_embedding, aggregation_last]
  trainable: none
teacher:
  family: sapiens2
  checkpoint: facebook/sapiens2-pretrain-0.1b
  input_hw: [1024, 768]
  layer: last
  frozen: true
data:
  name: synthetic
  split: test
  num_views: 2
  batch_size: 1
  max_samples: 10
correspondence:
  samples_per_pair: 1024
  occlusion_tolerance_m: 0.01
  seed: 42
logging:
  save_features: false
  save_visualizations: true
```
Define strict schema, CLI overrides, fail on unknown fields, expand env explicitly, save resolved config. For local fixture use mock encoders with explicit `mock: true`; no fake results presented as real. CLI target examples to implement:
```
python -m human_vggt.cli doctor
python -m human_vggt.cli prepare --config configs/experiment/e00_synthetic.yaml
python -m human_vggt.cli evaluate --config configs/experiment/e03_correspondence.yaml
python -m human_vggt.cli train --config configs/experiment/e04_distill.yaml
pytest -q
```
Every run writes outputs/<experiment>/<run_id>/{config.yaml,provenance.json,metrics.json,per_sample.csv,logs.txt,figures/,checkpoints/ as needed}. Record git SHA + dirty status, upstream SHAs, model revision, dataset manifest hash, seeds, environment, elapsed time, peak GPU memory, actual command. Never overwrite results; resume explicit. Use JSON/CSV + concise markdown summaries, optional private W&B only on opt-in.

## 8. First Codex session deliverables
1. Read AGENTS.md; inspect empty/existing directory; initialize Git without destroying anything. Create scaffold, ignore rules, manifests and upstream submodules. If clone network blocked, record exact error and continue.
2. Implement deterministic synthetic fixture, geometry utilities, tests and strict config; run CPU tests locally. Test resize/crop intrinsics, z-buffer occlusion, special-token stripping, transforms, known Sim(3), masked metrics and perfect/noisy cases.
3. Implement actual model wrappers against inspected upstream source; add optional heavy smoke tests. Unit tests use mock tensors; do not silently substitute mocks for checkpoints. Record shapes for actual models when available.
4. Implement baseline and E03 correspondence CLI, dataset adapter contract and minimal plots. Prepare THuman render plan, access checklist and remote script. Complete useful unblocked work even if external data unavailable.
5. Produce docs/experiments/e00_status.md with exact commands, tests actually run, files changed, blockers, assumptions, next runnable commands; commit coherent changes locally. Do not begin E04 full training before baseline/correspondence integrity verified; do not demand positive E03 results before implementing a small transfer pilot.

Definition of done: clean or explicitly explained git status; runnable local fixture; tests pass or failures recorded; real upstream adapters reviewed; remote instructions reproducible; no claims about GPU/data runs not performed. No server credentials or actual repository execution supplied in this brief.

## Sources (official; checked 2026-11-18)
- VGGT https://github.com/facebookresearch/vggt ; aggregator https://github.com/facebookresearch/vggt/blob/main/vggt/models/aggregator.py
- Sapiens 2 https://github.com/facebookresearch/sapiens2 ; checkpoint catalog https://github.com/facebookresearch/sapiens2/blob/main/docs/MODEL_ZOO.md
- DINOv2 https://github.com/facebookresearch/dinov2 ; Omega https://github.com/facebookresearch/vggt-omega
- SAM 3D Body https://github.com/facebookresearch/sam-3d-body
- THuman https://github.com/ytrock/THuman2.0-Dataset ; ICON https://github.com/YuliangXiu/ICON
- PKU-DyMVHumans https://pku-dymvhumans.github.io/ ; DNA-Rendering https://dna-rendering.github.io/ ; Hi4D https://yifeiyin04.github.io/Hi4D/ ; ETH3D https://www.eth3d.net/overview
