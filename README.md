# Human-VGGT

Research code for testing whether Sapiens 2 features transfer human surface
priors to an RGB-only VGGT reconstruction model. The roadmap defines the
baseline, correspondence experiments, and evaluation protocols. See
[RESEARCH_PLAN.md](RESEARCH_PLAN.md) and [E00 status](docs/experiments/e00_status.md).

## Apple Silicon

With Git and [uv](https://docs.astral.sh/uv/) installed, run from the repository root.

```sh
scripts/bootstrap.sh
scripts/doctor.sh
scripts/run.sh prepare --config configs/experiment/e00_synthetic.yaml
scripts/run.sh evaluate --config configs/experiment/e00_synthetic.yaml
scripts/run.sh evaluate --config configs/experiment/e03_correspondence.yaml
.venv/bin/pytest -q
```

The fixture runs on CPU with generated RGB, masks, optical-axis depth, surface
IDs, and calibrated OpenCV cameras. E00 uses known geometry to test the
evaluators. E03 uses two patch sizes of pooled RGB to exercise sampling and
retrieval. Both runs carry `fixture: true` and produce harness evidence.

## Linux NVIDIA

On a Linux x86-64 checkout, use a host driver compatible with the PyTorch CUDA
12.8 wheels. Bootstrap selects the CUDA lock on that platform. Initialize the
checkout with `git submodule update --init --recursive` and use
`git pull --ff-only` for updates. The repository has no configured remote.
An authorized private remote can carry the parent commit and gitlinks.
For uncommitted changes, `scripts/sync.sh user@host:/path/to/checkout/` previews
an rsync to an existing checkout. Pass `--apply` to send the previewed changes.

```sh
export HUMAN_VGGT_DATA=/path/to/calibrated/sample
export HUMAN_VGGT_WEIGHTS=/path/to/checkpoints
export HUMAN_VGGT_RUNS=/path/to/runs
scripts/bootstrap.sh
scripts/doctor.sh
```

Review the upstream licenses before using the corresponding code and
checkpoints. The primary VGGT research weights use CC-BY-NC-4.0. The Sapiens
checkpoint uses the Sapiens2 License. After any required authorization, place
the pinned files in the separate directories below. The adapters verify the
full SHA-256 before loading. [Pinned revisions and licenses](manifests/upstream/revisions.yaml)

```sh
.venv/bin/hf download facebook/VGGT-1B model.safetensors \
  --revision 860abec7937da0a4c03c41d3c269c366e82abdf9 \
  --local-dir "$HUMAN_VGGT_WEIGHTS/vggt"
scripts/run.sh evaluate --config configs/experiment/e00_vggt.yaml
HUMAN_VGGT_TEST_UPSTREAM=1 .venv/bin/pytest -q -m upstream -k vggt
```

E00 compares the adapter and upstream quick-start outputs on two RGBs and
exports pose encoding, K, world-to-camera poses, depth, direct world points,
depth-unprojected points, and both confidence maps. Two camera centers give a
degenerate Sim(3), so this configuration yields parity checks and
`alignment_unavailable` for metric scoring. Use `--set num_views=3` to try
the sequence camera-center alignment. On an ordered directory of RGBs, use
`--set dataset=rgb --set rgb_dir=/path/to/images` and set the exact view count.

After the E00 parity check, run a bounded representation smoke for E01/E03.

```sh
.venv/bin/hf download facebook/sapiens2-pretrain-0.1b sapiens2_0.1b_pretrain.safetensors \
  --revision b16fc30feddc2cf9330b2980a26a8cc91f51e697 \
  --local-dir "$HUMAN_VGGT_WEIGHTS/sapiens2"
scripts/run.sh evaluate --config configs/experiment/e03_upstream_smoke.yaml
HUMAN_VGGT_TEST_UPSTREAM=1 HUMAN_VGGT_TEST_SAPIENS=1 .venv/bin/pytest -q -m upstream
scripts/run.sh evaluate --config configs/data/e03_calibrated.yaml
```

## Calibrated input and results

The calibrated pilot evaluates a single synchronized sample. Its directory
contains a `sample.json` with `sample_id`, `scene_id`, `split`,
`provenance`, `depth_unit: "m"`, `depth_type: "z"`, and ordered `views`.
Supply `subject_id` for human results and `time_id` for dynamic captures.
Each view has `view_id`, `rgb_path`, `mask_path`, `depth_path`, `K`
(3×3), and `T_cw` (4×4). Relative paths resolve from the sample directory.
Masks are PNGs with nonzero foreground. Depth is an `[H,W]` NumPy `.npy`
array in meters, with zero for invalid or missing depth. RGB and GT must
share original pixel coordinates. Cameras use OpenCV axes and
integer pixel centers. Models receive RGB, and evaluators use calibration,
masks, and depth.

Each run creates a unique output directory with the resolved config, dataset
manifest and digest, Git and submodule revisions, package and device details,
model metadata, metrics, per-view or per-pair CSV, and a log. Baseline runs
include raw prediction arrays. Correspondence runs with valid pairs include a
recall plot.

Dependency rebuild commands are recorded in the headers of
[the Mac lock](env/mac.lock.txt) and [the Linux CUDA lock](env/linux-cu128.lock.txt).
The project uses the inference paths from the pinned submodules and avoids
installing the full upstream demo and training stacks.
