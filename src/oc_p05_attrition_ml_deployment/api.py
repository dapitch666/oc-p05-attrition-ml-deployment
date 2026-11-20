from fastapi import FastAPI

app = FastAPI(
    title="Futurisys attrition API",
    description="API serving the employee attrition classifier.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API is running."""
    return {"status": "ok"}
