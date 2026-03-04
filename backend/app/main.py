from fastapi import FastAPI

from app.routes.strategy import router as strategy_router

app = FastAPI(
    title="Options Analyzer",
    description="Options strategy payoff and metrics engine — Phase 1",
    version="0.1.0",
)

app.include_router(strategy_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
