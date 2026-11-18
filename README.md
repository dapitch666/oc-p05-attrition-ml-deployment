# Attrition model deployment

Deploys the employee attrition model from project 4 (CatBoost, alert threshold 0.25)
behind a FastAPI API, with PostgreSQL logging, Docker Compose and a GitHub Actions
CI/CD pipeline. Project 5 of the OpenClassrooms AI Engineer course.

## Stack

Python 3.12, uv, FastAPI, PostgreSQL, SQLAlchemy, Pytest, Docker Compose, GitHub Actions.

## Development setup

```bash
uv sync
uv run pre-commit install
```

## Conventions

- Branches: `<type>/<short-topic>` (`feat`, `fix`, `test`, `docs`, `ci`, `chore`, `refactor`).
- Commits: `type: description`. The type is added by a hook from the branch name.
- Changes reach `main` through a pull request with a green CI.
