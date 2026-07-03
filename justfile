set shell := ["bash", "-cu"]

# List available recipes
default:
    @just --list

# One-time local setup: activate committed git hooks + sync the locked env.
# pre-commit runs `just check`; pre-push runs `just validate`.
setup:
    git config core.hooksPath hooks
    uv sync

# Static gates: lint + format-check + type-check (fast, no game needed)
check:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy

# Auto-format and apply safe lint fixes
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# Automated tests (T1-T3, T5-T6; live smoke skips when capture is unavailable)
test:
    uv run pytest

# THE full CI action = check + test + accuracy eval. Green here => CI green.
validate: check test eval

# Accuracy-eval harness; model metrics report pending-model until MP4.
eval *ARGS:
    uv run zero-ad-eyes eval {{ARGS}}

# Run the live production capture; writes raw video, overlay video, and world models.
run *ARGS:
    uv run zero-ad-eyes run --live --record --record-overlay {{ARGS}}

# Run the pipeline and also mirror world models to stdout for debugging.
run-stdout *ARGS:
    uv run zero-ad-eyes run --stdout {{ARGS}}

# One-frame live smoke: exercises real capture + overlay without recording a long run.
smoke-live CONFIG="config.json":
    uv run zero-ad-eyes run --live --frames 1 --overlay --config {{CONFIG}}

# Continuous live calibration. Freeze each requested frame, then draw the ROI.
calibrate-live CONFIG="config.json":
    uv run zero-ad-eyes calibrate --live --config {{CONFIG}}

# Replay a recording and write world models to JSONL.
replay RECORDING CONFIG="config.json":
    uv run zero-ad-eyes run --recording {{RECORDING}} --config {{CONFIG}}

# Live demo artifact: raw recording + sibling overlay recording.
record-live CONFIG="config.json":
    uv run zero-ad-eyes run --live --record --record-overlay --config {{CONFIG}}

# Opt-in debug path for the old noisy classical main-viewport detector.
debug-classical RECORDING CONFIG="config.json":
    uv run zero-ad-eyes run --recording {{RECORDING}} --config {{CONFIG}} --detector classical

# Perf benchmark; provisional until MP4 and not part of validate.
bench *ARGS:
    uv run zero-ad-eyes bench {{ARGS}}
