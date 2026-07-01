set shell := ["bash", "-cu"]

# List available recipes
default:
    @just --list

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

# Accuracy-eval harness (ML8). Model-dependent (🔌) metrics report pending-model
# until the real adapter lands (MP4). See REQUIREMENTS.md §5.10.4.
eval:
    @echo "eval: NF3 model-dependent metrics = pending-model (no trained model yet)"

# Launch the perception layer (synthetic source + stub model until adapters land)
run *ARGS:
    uv run zero-ad-eyes run {{ARGS}}
