jupyter:
	uv run --with jupyter jupyter lab --no-browser

lint:
	uv run ruff check

fix:
	uv run ruff check --fix

format:
	uv run ruff format