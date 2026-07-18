# mitm-inspector convenience targets.
# Wraps the checked-in `uv run` / `npm` commands so common flows are one word.
# Backend at :8000 serves only the API; the SPA is served by vite (dev at :5173
# with HMR, preview at :4173 from web/dist). vite is configured to proxy
# /api/v1 (including the WebSocket stream) at both dev and preview.

PY_VERSION := 3.12
WEB_DIR    := web
APP_PORT   := 8000
PROXY_PORT := 8080
DEV_PORT   := 5173
PREVIEW_PORT := 4173

.PHONY: help up up-prod install install-py install-web build \
        api dev preview run plan test test-py test-web \
        gate gate-e2e ruff mypy pytest typecheck lint vitest e2e \
        clean clean-py clean-web clean-dist

help:
	@echo "targets:"
	@echo "  up          install + launch api + reverse proxy + vite dev SPA (HMR at :$(DEV_PORT))"
	@echo "  up-prod     install + build SPA + launch api + reverse proxy + vite preview (:$(PREVIEW_PORT))"
	@echo "  api         launch backend + reverse proxy only (no SPA server)"
	@echo "  dev         vite dev server only (needs api running separately)"
	@echo "  preview     vite preview of web/dist (needs build first)"
	@echo "  install     install-py + install-web"
	@echo "  build       build SPA into web/dist"
	@echo "  plan        print child argv without launching (dry-run)"
	@echo "  test        gate (pytest, ruff, mypy, typecheck, lint, vitest)"
	@echo "  gate-e2e    gate + Playwright real-browser responsive probe"
	@echo "  clean       remove .venv, web/node_modules, web/dist, caches"
	@echo ""
	@echo "urls after 'make up':"
	@echo "  UI    http://127.0.0.1:$(DEV_PORT)/     (vite dev, HMR)"
	@echo "  API   http://127.0.0.1:$(APP_PORT)/     (mitm-inspector backend)"
	@echo "  PROXY http://127.0.0.1:$(PROXY_PORT)/   (point ANTHROPIC_BASE_URL here)"

up: install
	@echo "spinning up: api :$(APP_PORT) | proxy :$(PROXY_PORT) | ui :$(DEV_PORT)"
	@echo "point anthropic client at:  http://127.0.0.1:$(PROXY_PORT)"
	@trap 'kill 0' EXIT INT TERM; \
	 ( cd $(WEB_DIR) && npm run dev -- --host 127.0.0.1 --port $(DEV_PORT) --strictPort ) & \
	 uv run --python $(PY_VERSION) mitm-inspector run --app-port $(APP_PORT) --proxy-port $(PROXY_PORT); \
	 wait

up-prod: install build
	@echo "spinning up (preview): api :$(APP_PORT) | proxy :$(PROXY_PORT) | ui :$(PREVIEW_PORT)"
	@trap 'kill 0' EXIT INT TERM; \
	 ( cd $(WEB_DIR) && npm run preview -- --host 127.0.0.1 --port $(PREVIEW_PORT) --strictPort ) & \
	 uv run --python $(PY_VERSION) mitm-inspector run --app-port $(APP_PORT) --proxy-port $(PROXY_PORT); \
	 wait

api:
	uv run --python $(PY_VERSION) mitm-inspector run --app-port $(APP_PORT) --proxy-port $(PROXY_PORT)

dev:
	cd $(WEB_DIR) && npm run dev -- --host 127.0.0.1 --port $(DEV_PORT) --strictPort

preview: build
	cd $(WEB_DIR) && npm run preview -- --host 127.0.0.1 --port $(PREVIEW_PORT) --strictPort

run: api

plan:
	uv run --python $(PY_VERSION) mitm-inspector plan --app-port $(APP_PORT) --proxy-port $(PROXY_PORT)

install: install-py install-web

install-py:
	uv sync --python $(PY_VERSION)

install-web:
	cd $(WEB_DIR) && npm ci

build: install-web
	cd $(WEB_DIR) && npm run build

ruff:
	uv run --python $(PY_VERSION) ruff check .

mypy:
	uv run --python $(PY_VERSION) mypy

pytest:
	uv run --python $(PY_VERSION) pytest -q

typecheck:
	cd $(WEB_DIR) && npm run typecheck

lint:
	cd $(WEB_DIR) && npm run lint

vitest:
	cd $(WEB_DIR) && npx vitest --run

e2e:
	cd $(WEB_DIR) && npm run test:e2e

test-py: ruff mypy pytest
test-web: typecheck lint vitest
test: test-py test-web
gate-e2e: test e2e

clean-py:
	rm -rf .venv .mypy_cache .pytest_cache .ruff_cache

clean-web:
	rm -rf $(WEB_DIR)/node_modules $(WEB_DIR)/tsconfig.node.tsbuildinfo

clean-dist:
	rm -rf $(WEB_DIR)/dist

clean: clean-py clean-web clean-dist
