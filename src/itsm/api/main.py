"""FastAPI application factory.

Backbone only: the routers delegate to the same library functions the CLI and
the notebooks use, so there is exactly one implementation of every behaviour.

What is deliberately in place:

* an **application factory** (``create_app``) so tests build an isolated app;
* a **lifespan** hook that prepares directories and logging, and never loads a
  model - artifacts load lazily on first use, keeping start-up instant;
* a **request id** on every response and in every error envelope, so a client
  report can be traced in the logs;
* **uniform error responses** (:class:`~itsm.api.schemas.ErrorResponse`).

What is deliberately left out, and would be added per deployment: authentication
(an API-key or OIDC dependency on the routers), rate limiting, and metrics
export. They belong to the platform, not to the model service.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from itsm import __version__
from itsm.api.routers import analytics, health, predict, privacy
from itsm.config import Settings, get_settings
from itsm.logging import get_logger, setup_logging

logger = get_logger("api")

DESCRIPTION = """
Serving layer for the ITSM insights project.

* **/v1/classify** - route a ticket to a category, with an abstention flag.
* **/v1/sentiment** - frustration polarity of a requester message.
* **/v1/clusters** - families discovered in the designations.
* **/v1/signals/scan** - spikes and emerging weak signals in ticket volumes.
* **/v1/stats/**, **/v1/tickets/{id}/trace** - lifecycle and status-update statistics.

Model endpoints answer **503** until the corresponding artifact has been
trained (`itsm train classifier`, `itsm analyze designations`).
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare the runtime; models stay lazy."""
    settings: Settings = get_settings()
    setup_logging(settings.log_level)
    settings.ensure_dirs()
    logger.info("starting %s v%s (dataset=%s)", settings.api_title, __version__, settings.dataset)
    yield
    logger.info("shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application."""
    settings = settings or get_settings()
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=DESCRIPTION,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "service", "description": "Health and model introspection."},
            {"name": "inference", "description": "Model predictions."},
            {"name": "analytics", "description": "Descriptive statistics and signals."},
            {"name": "privacy", "description": "Removal of personal data from free text."},
        ],
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def add_request_context(
        request: Request, call_next: Callable[[Request], Awaitable[JSONResponse]]
    ) -> JSONResponse:
        """Attach a request id and a server-timing header to every response."""
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["x-request-id"] = request_id
        response.headers["x-process-time-ms"] = f"{elapsed_ms:.1f}"
        logger.info(
            "%s %s -> %s in %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Return a uniform envelope instead of leaking a stack trace."""
        request_id = getattr(request.state, "request_id", None)
        logger.exception("unhandled error on %s (request_id=%s)", request.url.path, request_id)
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error", "request_id": request_id},
        )

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(analytics.router)
    app.include_router(privacy.router)
    return app


app = create_app()
