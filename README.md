# Attrition model deployment

Deploys the employee attrition model from project 4 (CatBoost, alert threshold 0.25)
behind a FastAPI API, with PostgreSQL logging, Docker Compose and a GitHub Actions
CI/CD pipeline. Project 5 of the OpenClassrooms AI Engineer course.

## Stack

Python 3.12, uv, FastAPI, PostgreSQL, SQLAlchemy, Pytest, Docker Compose, GitHub Actions.

## Installation

```bash
uv sync
cp .env.example .env
```

Fill in the values in `.env`. It is not versioned.

## Usage

To run the API on the host, start PostgreSQL with Docker Compose, then start uvicorn:

```bash
docker compose up -d db
uv run --env-file .env uvicorn oc_p05_attrition_ml_deployment.api:app
```

The API listens on `http://127.0.0.1:8000`. `GET /health` reports that it runs,
`POST /predict` takes the raw employee fields and returns the score, the alert flag and
the model version. The Swagger documentation is served at `/docs`. The tables are created
at startup, and every call is logged in PostgreSQL.

## Deployment

A `Dockerfile` builds the image in two stages with uv: the dependencies and the package
are installed in a virtual environment, then copied with the model into a slim Python
image that runs as a non-root user. `.dockerignore` lists the few files allowed in the
build context, so `data/` and `.env` never enter the image.

To run the API and PostgreSQL together in containers:

```bash
docker compose up --build
```

The `api` service waits for the `db` health check, reaches the database at host `db`
and is bound to `127.0.0.1:8000`. `APP_ENV` and the `POSTGRES_*` variables come from `.env`.

The `publish` workflow builds the image for `linux/amd64` when a `v*` tag is pushed and
publishes it to `ghcr.io/dapitch666/oc-p05-attrition-ml-deployment`, tagged with the
version. It fails when the tag differs from the version in `pyproject.toml` or when the
tagged commit is not on `main`. Run manually, it builds the image without pushing.

To run the published image instead of building it, write a Compose override, for example
`docker-compose.ghcr.yml`:

```yaml
services:
  api:
    image: ghcr.io/dapitch666/oc-p05-attrition-ml-deployment:0.4.0
    pull_policy: always
```

Then start the stack with the override and without building:

```bash
docker compose -f docker-compose.yml -f docker-compose.ghcr.yml up -d --no-build api
```

The image is public, so no login is needed. Version 0.4.0 was pulled and run this way,
and the `/health` and `/predict` endpoints were checked.

## Project tree

```text
.
├── .dockerignore
├── .env.example
├── .github/workflows/
│   ├── ci.yml
│   └── publish.yml
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── Dockerfile
├── README.md
├── docker-compose.yml
├── models/
│   ├── attrition_metadata.json
│   └── attrition_pipeline.joblib
├── pyproject.toml
├── scripts/
│   ├── create_db.py
│   ├── prepare_commit_msg.py
│   └── train_model.py
├── src/oc_p05_attrition_ml_deployment/
│   ├── __init__.py
│   ├── api.py
│   ├── dataset.py
│   ├── db.py
│   ├── features.py
│   ├── model.py
│   ├── models.py
│   └── schemas.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_dataset.py
│   ├── test_db.py
│   ├── test_features.py
│   └── test_model.py
└── uv.lock
```

## Development setup

```bash
uv sync
uv run pre-commit install
```

## Conventions

- Branches: `<type>/<short-topic>` (`feat`, `fix`, `test`, `docs`, `ci`, `chore`, `refactor`).
- Commits: `type: description`. The type is added by a hook from the branch name.
- Changes reach `main` through a pull request with a green CI.
