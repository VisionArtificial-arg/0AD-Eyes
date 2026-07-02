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

# Automated tests (T1-T3, T5-T6)
test:
    uv run pytest

# THE full CI action = check + test + accuracy eval. Green here => CI green.
validate: check test eval

# Accuracy-eval harness (ML8). Runs the real evaluate() harness; model-dependent
# (🔌) metrics report pending-model until the real adapter lands (MP4). Pass a
# dataset with `just eval --dataset PATH`. See REQUIREMENTS.md §5.10.4.
eval *ARGS:
    uv run zero-ad-eyes eval {{ARGS}}

# Launch the perception layer (synthetic source + stub model until adapters land)
run *ARGS:
    uv run zero-ad-eyes run {{ARGS}}

# Perf benchmark (T6/NF1/NF2): latency percentiles + throughput. Provisional on the
# stub/classical path; the real NF1 gate closes at MP4. Not in `validate` (timing
# gates are machine-dependent); the harness logic is covered by pytest instead.
bench *ARGS:
    uv run zero-ad-eyes bench {{ARGS}}
