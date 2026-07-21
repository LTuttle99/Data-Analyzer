import logging
import os
import traceback
import uuid
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from analyzer import BookOfBusinessAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_analyzer")

# Set DEBUG=1 in the environment to include server tracebacks in API error responses.
# Leave unset in production so internal stack traces aren't exposed to clients.
DEBUG_MODE = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")

app = FastAPI(title="Intelligent Data Analyzer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SESSION_COOKIE_NAME = "analyzer_session_id"
MAX_SESSIONS = 100  # simple cap so a long-running server doesn't grow unbounded

# Two independent per-session slots: the primary uploaded file, and an optional
# comparison snapshot (a second file) used for period-over-period comparisons.
PRIMARY_CACHE: dict[str, BookOfBusinessAnalyzer] = {}
COMPARE_CACHE: dict[str, BookOfBusinessAnalyzer] = {}


# --------------------------------------------------------------------------- #
# Request schemas
# --------------------------------------------------------------------------- #

class ColumnValuesRequest(BaseModel):
    column: Optional[str] = None


class DateRangeRequest(BaseModel):
    timeline_column: Optional[str] = None


class GoalConfig(BaseModel):
    id: Optional[str] = None
    label: Optional[str] = "Goal"
    period: str = "annual"
    scope_type: str = "overall"          # "overall" | "dimension"
    scope_column: Optional[str] = None   # any mapped dimension column, when scope_type == "dimension"
    scope_value: Optional[str] = None
    target_value: float = 0.0


class MappingConfig(BaseModel):
    metric_column: Optional[str] = None
    timeline_column: Optional[str] = None
    entity_column: Optional[str] = None
    dimension_columns: list = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    mapping: MappingConfig
    dimension_filters: dict = Field(default_factory=dict)  # {column_name: [allowed values]}
    projection_target: str = "value"     # "value" (sum of metric_column) or "count" (unique entities)
    primary_dimension: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    include_future_dates: bool = False
    goal_value: float = 0.0
    goals: list[GoalConfig] = Field(default_factory=list)
    target: str = "primary"  # "primary" or "compare" — which uploaded file to analyze
    forecast_horizon_months: int = 24
    entity_view: str = "all"  # "all" | "new" | "repeat"


class SuggestGoalsRequest(BaseModel):
    mapping: MappingConfig
    projection_target: str = "value"
    period: str = "annual"
    top_n: int = 3


# --------------------------------------------------------------------------- #
# Session helpers
#
# IMPORTANT: FastAPI only auto-merges cookies/headers set on an injected
# `response: Response` parameter into the final response when the endpoint
# returns something FastAPI has to serialize itself (a dict/model). If the
# endpoint constructs and returns its own Response/JSONResponse directly —
# which every route here does — that merge never happens and the cookie is
# silently dropped. So instead, every route calls `stamp_session_cookie()` on
# the *actual* response object it returns, in both success and error paths.
# --------------------------------------------------------------------------- #

def get_session_id(request: Request) -> str:
    """Read the existing session cookie, or mint a new one for this request."""
    return request.cookies.get(SESSION_COOKIE_NAME) or uuid.uuid4().hex


def stamp_session_cookie(resp: Response, session_id: str) -> Response:
    """Attach the session cookie directly to the response object being returned."""
    resp.set_cookie(
        SESSION_COOKIE_NAME,
        session_id,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24
    )
    return resp


def _store_in_cache(cache: dict, session_id: str, analyzer: BookOfBusinessAnalyzer) -> None:
    """Insert into a session cache, evicting the oldest entry once the cap is hit."""
    if session_id not in cache and len(cache) >= MAX_SESSIONS:
        oldest_key = next(iter(cache))
        cache.pop(oldest_key, None)

    cache[session_id] = analyzer


def get_active_analyzer(session_id: str, target: str = "primary") -> BookOfBusinessAnalyzer:
    """Fetch the analyzer for this session/slot, or raise a clear 400 error."""
    cache = COMPARE_CACHE if target == "compare" else PRIMARY_CACHE
    analyzer = cache.get(session_id)

    if not analyzer:
        label = "comparison" if target == "compare" else "primary"
        raise HTTPException(
            status_code=400,
            detail=f"No active {label} data file found for this session. Upload a file first."
        )

    return analyzer


def error_response(exc: Exception, status_code: int = 500) -> JSONResponse:
    """Log the full traceback server-side and return a clean, bounded error to the client."""
    logger.error("Request failed: %s", exc, exc_info=True)

    message = str(exc) or exc.__class__.__name__

    if DEBUG_MODE:
        message = f"{message}\n\nTraceback:\n{traceback.format_exc()}"

    return JSONResponse(status_code=status_code, content={"error": message})


def _model_to_dict(model: BaseModel) -> dict:
    """Support both pydantic v2 (model_dump) and v1 (dict) without pinning a version."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Two things this fixes vs. FastAPI's default handler:
    1. Returns {"error": ...} instead of {"detail": ...}, matching every other error path
       in this app, so the frontend's error-message extraction actually finds it.
    2. Stamps the session cookie even on error responses, so a session established on a
       failing request still works correctly on the next one.
    """
    session_id = get_session_id(request)
    resp = JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
    return stamp_session_cookie(resp, session_id)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/api/health")
async def health_check(request: Request):
    session_id = get_session_id(request)

    resp = JSONResponse(content={
        "status": "ok",
        "service": "Intelligent Data Analyzer",
        "primary_active": session_id in PRIMARY_CACHE,
        "compare_active": session_id in COMPARE_CACHE
    })

    return stamp_session_cookie(resp, session_id)


@app.post("/api/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    session_id = get_session_id(request)

    try:
        contents = await file.read()

        if not contents:
            raise ValueError("The uploaded file is empty.")

        analyzer = BookOfBusinessAnalyzer(contents, file.filename)
        schema = analyzer.infer_schema()

        _store_in_cache(PRIMARY_CACHE, session_id, analyzer)

        # A fresh primary upload invalidates any prior comparison snapshot for this session,
        # since it was likely set up to compare against the previous primary file.
        COMPARE_CACHE.pop(session_id, None)

        resp = JSONResponse(content=schema)

    except Exception as e:
        resp = error_response(e, status_code=400)

    return stamp_session_cookie(resp, session_id)


@app.post("/api/compare-upload")
async def upload_compare_file(request: Request, file: UploadFile = File(...)):
    """Upload a second file (e.g. last month's export) to compare against the primary one."""
    session_id = get_session_id(request)

    try:
        contents = await file.read()

        if not contents:
            raise ValueError("The uploaded comparison file is empty.")

        analyzer = BookOfBusinessAnalyzer(contents, file.filename)
        schema = analyzer.infer_schema()

        _store_in_cache(COMPARE_CACHE, session_id, analyzer)

        resp = JSONResponse(content=schema)

    except Exception as e:
        resp = error_response(e, status_code=400)

    return stamp_session_cookie(resp, session_id)


@app.post("/api/column-values")
async def column_values(body: ColumnValuesRequest, request: Request):
    session_id = get_session_id(request)

    try:
        analyzer = get_active_analyzer(session_id)
        resp = JSONResponse(content={
            "values": analyzer.get_unique_column_values(body.column)
        })

    except HTTPException:
        raise
    except Exception as e:
        resp = error_response(e)

    return stamp_session_cookie(resp, session_id)


@app.post("/api/date-range")
async def refresh_date_range(body: DateRangeRequest, request: Request):
    session_id = get_session_id(request)

    try:
        analyzer = get_active_analyzer(session_id)
        resp = JSONResponse(content=analyzer.get_date_range(body.timeline_column))

    except HTTPException:
        raise
    except Exception as e:
        resp = error_response(e)

    return stamp_session_cookie(resp, session_id)


@app.post("/api/analyze")
async def analyze_data(body: AnalyzeRequest, request: Request):
    session_id = get_session_id(request)

    try:
        target = "compare" if body.target == "compare" else "primary"
        analyzer = get_active_analyzer(session_id, target=target)

        results = analyzer.run_analysis(
            mapping=_model_to_dict(body.mapping),
            dimension_filters=body.dimension_filters,
            projection_target=body.projection_target,
            primary_dimension=body.primary_dimension,
            start_date=body.start_date,
            end_date=body.end_date,
            include_future_dates=body.include_future_dates,
            goal_value=body.goal_value,
            goals=[_model_to_dict(g) for g in body.goals],
            forecast_horizon_months=body.forecast_horizon_months,
            entity_view=body.entity_view
        )

        resp = JSONResponse(content=results)

    except HTTPException:
        raise
    except Exception as e:
        resp = error_response(e)

    return stamp_session_cookie(resp, session_id)


@app.post("/api/suggest-goals")
async def suggest_goals(body: SuggestGoalsRequest, request: Request):
    session_id = get_session_id(request)

    try:
        analyzer = get_active_analyzer(session_id)

        suggestions = analyzer.suggest_goal_candidates(
            mapping=_model_to_dict(body.mapping),
            projection_target=body.projection_target,
            period=body.period,
            top_n=body.top_n
        )

        resp = JSONResponse(content={"suggestions": suggestions})

    except HTTPException:
        raise
    except Exception as e:
        resp = error_response(e)

    return stamp_session_cookie(resp, session_id)


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    template_path = os.path.join(BASE_DIR, "templates", "index.html")

    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Template index.html missing from repository layout structure."
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
