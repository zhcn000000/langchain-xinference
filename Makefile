.PHONY: all format lint test tests integration_tests docker_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
integration_test integration_tests: TEST_FILE = tests/integration_tests/

PARENT_COMMIT := $(shell git rev-parse --verify HEAD^ 2>/dev/null || echo "4b825dc642cb6eb9a060e54bf8d69288fbee4904")
CHANGED_FILES := $(shell git diff --name-only $(PARENT_COMMIT)..$(GITHUB_SHA))

# unit tests are run with the --disable-socket flag to prevent network calls
test tests:
	pytest --disable-socket --allow-unix-socket $(TEST_FILE)

test_watch:
	ptw --snapshot-update --now . -- -vv $(TEST_FILE)

# integration tests are run without the --disable-socket flag to allow network calls
integration_test integration_tests:
	pytest $(TEST_FILE)

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/partners/xinference --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_xinference
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test


lint:
ifneq ($(CHANGED_FILES),)
	@echo "$(CHANGED_FILES)" | tr ' ' '\n'

	@echo "$(CHANGED_FILES)" | tr ' ' '\n' | grep -E '\.py$$|\.ipynb$$' | xargs -r ruff check

	@echo "$(CHANGED_FILES)" | tr ' ' '\n' | grep -E '\.py$$|\.ipynb$$' | xargs -r ruff format --diff

else
	@echo "没有检测到文件变更，跳过检查"
endif

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	codespell --toml pyproject.toml

spell_fix:
	codespell --toml pyproject.toml -w

check_imports: $(shell find langchain_xinference -name '*.py')
	python ./scripts/check_imports.py $^

######################
# HELP
######################

help:
	@echo '----'
	@echo 'check_imports				- check imports'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
