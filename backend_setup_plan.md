# Backend Setup Plan - Phase 1

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Establish the initial FastAPI backend scaffolding described in `react-refactoring-plan.md`.

## Mid-Level Objective

- Create the `backend/` directory tree with routers, services, and models subfolders.
- Provide a `main.py` entry point exposing a `/health` endpoint.
- Move `utils/llm_manager.py` into `backend/services/` and create placeholder service modules.
- Add empty Pydantic model files under `backend/models/`.
- Include a `dependencies.py` module and `requirements.txt` listing core packages.
- Document how to start the backend in the root `README.md`.

## Implementation Notes
- Keep the existing `main.py` at repository root for reference.
- All new modules should contain minimal placeholder content or TODO markers.
- Follow PEP8 style for Python files.
- Use FastAPI 0.110+, Pydantic 2+, and Uvicorn.

## Context

### Beginning context
- `README.md`
- `main.py`
- `requirements.txt`
- `utils/llm_manager.py`

### Ending context
- `backend/` with `main.py`, `routers/`, `services/`, `models/`, `dependencies.py`, and `requirements.txt`
- Updated `README.md` describing backend startup

## Low-Level Tasks
> Ordered from start to finish

1. Create backend directories and placeholder files
```aider
Use bash commands to make directories `backend`, `backend/routers`, `backend/services`, `backend/models`.
Create empty files for routers (`practice.py`, `evaluation.py`, `assistance.py`, `languages.py`, `__init__.py`) each with a placeholder APIRouter.
Create service files `llm_manager.py` (moved from utils), `language_support.py`, `code_execution.py` containing TODO comments.
Create model files `__init__.py`, `schemas.py`, `prompts.py`, `languages.py` with TODOs.
Create empty `dependencies.py` and `requirements.txt` listing FastAPI packages.
```
2. Implement `backend/main.py`
```aider
Create `backend/main.py` defining a FastAPI app that includes routers and exposes `/health` returning `{"status": "ok"}`.
```
3. Update documentation
```aider
Update `README.md` with a section explaining how to install backend dependencies and run the FastAPI app using `uvicorn backend.main:app`.
```
