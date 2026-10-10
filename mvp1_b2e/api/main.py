from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .db import engine
from .models import Base
from .routes import workspaces, sources, jobs


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup (dev only; production uses Alembic migrations)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(
    title="Pact Enterprise API",
    version="0.1.0",
    description="Decision intelligence for enterprise teams.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

PREFIX = "/api/v1"
app.include_router(workspaces.router, prefix=PREFIX)
app.include_router(sources.router, prefix=PREFIX)
app.include_router(jobs.router, prefix=PREFIX)


@app.get("/health")
async def health():
    return {"status": "ok", "company": settings.company_name}
