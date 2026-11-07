import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, closing
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .catalog import router as catalog_router
from .db import connect, database_path, initialize_database
from .workflow import router as workflow_router

STATIC_DIR = Path(__file__).with_name("static")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    path = database_path()
    initialize_database(path)
    app.state.database_path = path
    yield


app = FastAPI(title="CQA Backend", lifespan=lifespan)
app.include_router(catalog_router)
app.include_router(workflow_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
    # Raw input can contain values JSON cannot serialize (for example a parsed infinity).
    def clean(value: str) -> str:
        return value.encode("utf-8", errors="replace").decode("utf-8")

    detail = [
        {
            "loc": [clean(part) if isinstance(part, str) else part for part in item["loc"]],
            "msg": clean(item["msg"]),
            "type": item["type"],
        }
        for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    try:
        with closing(connect(request.app.state.database_path)) as connection:
            connection.execute("PRAGMA user_version").fetchone()
    except sqlite3.Error as error:
        raise HTTPException(status_code=503, detail="Database unavailable") from error

    return {"status": "ok", "database": "ok"}
