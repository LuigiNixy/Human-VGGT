# AGENTS.md — Human-VGGT

## Mission and authority
Read RESEARCH_PLAN.md before implementation. This is a research codebase for human-centric feature transfer into generic VGGT reconstruction. Keep the final model free-form, RGB-only, and independent of Sapiens/SMPL/MHR at inference. Do not quietly change the scientific question. Negative results are useful. Follow explicit user instructions over defaults here.

## Work autonomously; avoid defensive paralysis
- Implement the simplest viable experiment, run the smallest meaningful test, inspect output, then iterate. Do not turn a coding request into a plan-only response, ask for permission for ordinary local edits, or wait for perfect requirements.
- Make reversible assumptions for missing noncritical details; record them. Use documented defaults and configuration rather than asking the user about every dependency or layer. Prefer a simple linear projector before elaborate fusion.
- When missing GPU, data, access, or network, continue with fixtures, adapters, tests and runnable remote commands. Do not invent data or represent a mock run as evidence. Clearly distinguish local tests from real GPU results.
- A failure should generate a concrete diagnosis and next experiment, not disable the project. Do not overengineer guardrails, framework abstractions, or error handling; validate critical shapes and coordinate conventions, then proceed. Do not interpret anti-defensive as permission to conceal risks, bypass licensing, or spend resources.
- No need to ask before routine reads, local non-destructive edits, dependency installation inside the project environment, bounded CPU tests, or local Git commits. Ask before destructive operations, privileged/system changes, publishing/pushing to a new remote, licensing acceptance on user's behalf, paid resources, or large remote training jobs without a budget. If an existing user-approved remote and budget are provided, operate within those bounds.

## Research integrity
- Keep primary hypotheses, baseline and splits fixed; label exploratory results. Do not tune on test or silently remove hard examples. Separate diagnostic feature results from geometry evidence.
- Use original upstream RGB transforms and actual pretrained weights for baseline; compare same images, camera tuples, seeds, masks and alignment protocol. Do not assume Sapiens and VGGT share a patch grid. Trace layers and strip real special tokens. Train teacher in no circumstance; frozen eval/no-grad.
- Log experiments and provenance. Never fabricate scores, timings, GPU access, checkpoint validity, dataset contents or successful tests. Preserve failures and negative controls. Use subject-level splits and statistics.
- Do not introduce body parameters into the proposed model. Masks are supervision/evaluation only unless a clearly labeled ablation; no GT cameras at inference for primary comparison. Keep visible-surface vs complete-mesh metrics separate. See protocol in RESEARCH_PLAN.md.

## Source tree and code rules
- src/human_vggt/models: upstream wrappers and projector; data: dataset schema/adapters/transforms; geometry: cameras, visibility, coordinate alignment; losses: explicit objective terms; evaluation: metrics/correspondence; cli.py and train.py: orchestration only.
- configs/: versioned model/data/experiment configs. scripts/: thin shell launchers only. tests/: CPU unit tests and opt-in integration tests. manifests/: pinned sources/revisions/splits. docs/: decisions, experiment results, status. third_party/: pinned Git submodules, read-only by default. env/: platform inputs and locks.
- data/, checkpoints/, cache/, outputs/, .venv/, .env and local secrets are ignored. No absolute user-specific paths. Resolve external paths from env/config. Keep per-run outputs immutable. Use small synthetic fixtures generated in tests, not raw scans in Git.
- Python 3.12, type hints for public interfaces, clear shape contracts, simple typed config validation, pytest, ruff. Reuse shared preprocessing and geometry. Avoid gratuitous classes, dependencies, wrappers, and blanket exception handlers. Use meaningful errors, explicit capabilities and optional extras. Do not hardcode CUDA or silently downgrade GPU runs to CPU; auto device only when explicitly configured.
- Preserve existing user changes; inspect before editing. Minimize unrelated refactors. Use a small patch and test rather than rewrite upstream. If upstream patch unavoidable, version it explicitly and note why.

## Git workflow
- Inspect `git status --short`, `git branch --show-current`, `git diff` before work. If empty and not in Git, initialize project root with `git init -b main`. Don't create nested repo accidentally. Do not overwrite user work.
- Use coherent commits with descriptive messages (e.g. `feat: add camera and reprojection utilities`, `test: validate feature coordinate mapping`). Stage explicit paths, review `git diff --cached`; avoid blind `git add .`. Commit completed logical milestones after tests. Track .gitmodules and pinned gitlinks, not copied upstream trees. Record submodule SHAs via `git submodule status --recursive`.
- Use current branch unless task warrants a new `exp/e03-correspondence` branch or user specified otherwise; do not switch away from dirty work. No force push, reset --hard, clean -fd, rewriting history, deleting branches or changing remote without explicit authorization. Never commit credentials, restricted data, checkpoints or large generated assets. No unsolicited push. Use private authorized remote and pull --ff-only on server; sync submodules after pull.

## Local and remote execution
- Local Mac: CPU fixture/tests mandatory when environment permits; MPS optional, no CUDA-only installs. Remote Linux: explicit CUDA, driver-compatible Torch, matching torchvision, platform lock. Run doctor before expensive jobs; use bounded smoke to validate memory, precision, gradients and outputs. No assumed host/account/partition. Do not report remote execution unless actually performed.
- Keep training configs deterministic and record seed, actual environment, commit, checkpoint revisions, data hash, resolved config, metrics and failures. Full runs require user-approved resources; if inaccessible, produce exact commands and continue local work. Store results in unique outputs/<experiment>/<run_id>.

## Definition of done and communication
- For implementation tasks, deliver code plus actual test outcomes, not just suggestions. Verify targeted tests first, broader suite when feasible. State unresolved errors with reproducing commands and practical next action. No fake placeholder implementations advertised as complete.
- End each milestone with what changed, commands/tests run, evidence, blockers, next experiment and commit SHA if committed. Keep docs/experiments updated. Prefer concise, technical responses; do not spend time repeatedly summarizing obvious steps.
- Read `.agents/skills/research-execution/SKILL.md` and `.agents/skills/experiment-integrity/SKILL.md` when present. If absent, follow equivalent rules above; create the supplied skill files when bootstrapping.
