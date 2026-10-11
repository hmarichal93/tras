ifneq ($(OS),Windows_NT)
	# On Unix-based systems, use ANSI codes
	BLUE = \033[36m
	BOLD_BLUE = \033[1;36m
	BOLD_GREEN = \033[1;32m
	RED = \033[31m
	YELLOW = \033[33m
	BOLD = \033[1m
	NC = \033[0m
endif

escape = $(subst $$,\$$,$(subst ",\",$(subst ',\',$(1))))

define exec
	@echo "$(BOLD_BLUE)$(call escape,$(1))$(NC)"
	@$(1)
endef

help:
	@echo "$(BOLD_GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-].+:.*?# .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?# "}; \
		{printf "  $(BOLD_BLUE)%-20s$(NC) %s\n", $$1, $$2}'

PACKAGE_NAME:=tras
# Modules held to the strict gate (format + lint + types). Add a module here once it is
# clean; the legacy GUI code is migrated gradually.
STRICT_PATHS:=tras/params.py tras/api.py tras/analysis tras/io tras/evaluation tras/cores \
	tras/cli/benchmark.py tras/cli/core.py tras/cli/detect.py tras/cli/export_json.py tras/utils/cli_config.py \
	tras/utils/batch_helper.py tras/utils/ring_properties.py tras/utils/ring_sampling.py \
	tras/utils/pos_exporter.py tras/utils/shape_rings.py tras/widgets/core_dialog.py tests

setup:  # Setup the development environment
	$(call exec,uv sync --dev)

format:  # Format code
	$(call exec,uv run ruff format)
	$(call exec,uv run ruff check --fix)

lint:
	$(call exec,uv run ruff format --check)
	$(call exec,uv run ruff check)

mypy:
	$(call exec,uv run mypy --package $(PACKAGE_NAME))

lint_strict:  # Format, lint and type-check the strict modules (what CI enforces)
	$(call exec,uv run ruff format --check $(STRICT_PATHS))
	$(call exec,uv run ruff check $(STRICT_PATHS))
	$(call exec,uv run mypy $(filter-out tests,$(STRICT_PATHS)))

check_translate: update_translate
	$(call exec,git diff --exit-code tras/translate)

check: lint mypy check_translate # Run checks

test:  # Run tests (headless; GUI tests use the offscreen Qt platform)
	$(call exec,uv run pytest -v tests/)

benchmark_cores:  # Core detector benchmark: make benchmark_cores URUDENDRO=<folder>
	$(call exec,uv run python -m tras.cli.benchmark cores $(URUDENDRO))

build:  # Build the package
	$(call exec,uv build)

update_translate:
	$(call exec,uv run --no-sync tools/update_translate.py)
