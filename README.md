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
the model version. It needs an API key in the `X-API-Key` header. The Swagger
documentation is served at `/docs`. The tables are created at startup, and every call is
logged in PostgreSQL.

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

## Authentication and security

`POST /predict` requires an API key in the `X-API-Key` header. `GET /health` stays open.
A missing, unknown or revoked key gets a 401 with the message `Invalid or missing API key.`
and a `WWW-Authenticate: APIKey` header. If the database cannot be reached while the key
is checked, the answer is a 503. In Swagger (`/docs`), the Authorize button takes the key.

Keys are created from the host, with the database port published by Compose, because
`scripts/` is not in the image. Start the database, then create a key for a client:

```bash
docker compose up -d db
uv run --env-file .env python scripts/manage_api_keys.py create "Client name"
```

The script prints the key id and the key. The key is shown once and cannot be recovered.
Call the API with it:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "X-API-Key: <key>" -H "Content-Type: application/json" \
  -d '<employee in the format of the Swagger example>'
```

To revoke a key, run `scripts/manage_api_keys.py revoke <id>`. To find an id, list the keys
without their hashes:

```sql
SELECT id, client_name, created_at, revoked_at FROM api_keys;
```

Only the SHA-256 hash of a key is stored, never the key. Each row of `model_inputs` holds
the id of the key that made the call (`api_key_id`, empty for calls logged before keys
existed). Secrets come from the environment: `.env` is not versioned, and `.env.example`
lists the variables to fill in.

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
│   ├── manage_api_keys.py
│   ├── prepare_commit_msg.py
│   └── train_model.py
├── src/oc_p05_attrition_ml_deployment/
│   ├── __init__.py
│   ├── api.py
│   ├── auth.py
│   ├── dataset.py
│   ├── db.py
│   ├── features.py
│   ├── model.py
│   ├── models.py
│   └── schemas.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_auth.py
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
