from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from config import Settings, get_settings
from routes import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = get_settings()

    app.state.settings = settings
    app.state.app_started = True

    yield


def normalize_cors_origins(origins: Any | None = None) -> list[str]:
    """
    Safe CORS setup for local React/Vite development.
    Handles list, tuple, set, comma-separated string, or empty value.
    """
    default_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:5176",
        "http://127.0.0.1:5176",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    if origins is None:
        return default_origins

    if isinstance(origins, str):
        custom_origins = [
            item.strip().rstrip("/")
            for item in origins.split(",")
            if item.strip()
        ]
        return list(dict.fromkeys(default_origins + custom_origins))

    if isinstance(origins, (list, tuple, set)):
        custom_origins = [
            str(item).strip().rstrip("/")
            for item in origins
            if str(item).strip()
        ]
        return list(dict.fromkeys(default_origins + custom_origins))

    return default_origins


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="2.0.0",
        description=(
            "Automated FireDesign-style feasibility package generator for "
            "sprinklers, fire alarm, HVAC, electrical and plumbing review layouts."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=normalize_cors_origins(settings.cors_origins),
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    frontend_dist = settings.frontend_dist
    frontend_assets = frontend_dist / "assets"

    if frontend_assets.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(frontend_assets)),
            name="assets",
        )

    @app.get("/health", tags=["System"])
    def health_check() -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": "online",
            "service": settings.app_name,
            "environment": settings.environment,
            "version": "2.0.0",
        }

    @app.get("/api/health", tags=["System"])
    def api_health_check() -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": "online",
            "service": settings.app_name,
            "environment": settings.environment,
            "version": "2.0.0",
            "accuracy_mode": {
                "ocr_enabled": getattr(settings, "enable_ocr", True),
                "known_templates_enabled": getattr(settings, "enable_known_test_templates", True),
                "auto_scale_enabled": getattr(settings, "enable_auto_scale", True),
            },
        }

    @app.get("/api/runtime", tags=["System"])
    def api_runtime() -> dict[str, Any]:
        """
        Runtime status endpoint.

        Note:
        /api/status is handled by backend/routes.py.
        Keeping this separate avoids duplicate route conflicts.
        """
        return {
            "status": "ok",
            "backend": "online",
            "service": settings.app_name,
            "environment": settings.environment,
            "frontend_dist_exists": frontend_dist.exists(),
            "frontend_assets_exists": frontend_assets.exists(),
            "accepted_files": sorted(settings.allowed_extensions),
            "max_upload_mb": settings.max_upload_mb,
            "cors_origins": normalize_cors_origins(settings.cors_origins),
            "docs": "/docs",
            "api_status": "/api/status",
            "api_analyze": f"{settings.api_prefix}/analyze",
        }

    app.include_router(router)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def root() -> Response:
        index_file = frontend_dist / "index.html"

        if index_file.exists():
            return FileResponse(str(index_file))

        return HTMLResponse(
            f"""
            <!doctype html>
            <html lang="en">
              <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>{settings.app_name}</title>
                <style>
                  :root {{
                    color-scheme: light;
                    --bg: #f4f7f6;
                    --card: #ffffff;
                    --border: #d9e2df;
                    --text: #172026;
                    --muted: #5f6f6b;
                    --brand: #0f766e;
                    --brand-dark: #115e59;
                  }}

                  * {{
                    box-sizing: border-box;
                  }}

                  body {{
                    margin: 0;
                    font-family: Arial, sans-serif;
                    color: var(--text);
                    background:
                      radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 34%),
                      linear-gradient(135deg, #f4f7f6 0%, #eef7f4 100%);
                  }}

                  main {{
                    max-width: 900px;
                    margin: 10vh auto;
                    padding: 24px;
                  }}

                  section {{
                    background: var(--card);
                    border: 1px solid var(--border);
                    border-radius: 18px;
                    padding: 32px;
                    box-shadow: 0 18px 50px rgba(0,0,0,0.08);
                  }}

                  h1 {{
                    margin: 0 0 10px;
                    font-size: 34px;
                    letter-spacing: -0.04em;
                  }}

                  p {{
                    line-height: 1.65;
                    font-size: 16px;
                    color: var(--muted);
                  }}

                  a {{
                    color: var(--brand);
                    font-weight: 700;
                    text-decoration: none;
                  }}

                  a:hover {{
                    color: var(--brand-dark);
                    text-decoration: underline;
                  }}

                  code {{
                    background: #eef3f1;
                    padding: 4px 7px;
                    border-radius: 6px;
                    color: #0f3f3b;
                  }}

                  ul {{
                    line-height: 1.9;
                    padding-left: 22px;
                  }}

                  .badge {{
                    display: inline-block;
                    background: #e7f7f2;
                    color: #0f766e;
                    border: 1px solid #b8e7d8;
                    border-radius: 999px;
                    padding: 6px 12px;
                    font-size: 13px;
                    font-weight: 700;
                    margin-bottom: 14px;
                  }}

                  .grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
                    gap: 14px;
                    margin-top: 20px;
                  }}

                  .box {{
                    border: 1px solid var(--border);
                    border-radius: 12px;
                    padding: 14px;
                    background: #fbfefd;
                  }}

                  .box strong {{
                    display: block;
                    margin-bottom: 5px;
                  }}
                </style>
              </head>
              <body>
                <main>
                  <section>
                    <span class="badge">Backend Online</span>
                    <h1>{settings.app_name}</h1>
                    <p>
                      The FastAPI backend is running successfully. Start the React frontend using:
                    </p>
                    <p>
                      <code>cd frontend</code><br>
                      <code>npm run dev</code>
                    </p>

                    <div class="grid">
                      <div class="box">
                        <strong>Health</strong>
                        <a href="/health">/health</a><br>
                        <a href="/api/health">/api/health</a>
                      </div>

                      <div class="box">
                        <strong>API</strong>
                        <a href="/api/status">/api/status</a><br>
                        <a href="/api/runtime">/api/runtime</a>
                      </div>

                      <div class="box">
                        <strong>Docs</strong>
                        <a href="/docs">/docs</a><br>
                        <a href="/redoc">/redoc</a>
                      </div>
                    </div>

                    <p>
                      Accuracy note: PNG/JPG analysis depends on OCR and image clarity.
                      DXF with clean room, wall, door and MEP layers gives the highest accuracy.
                    </p>
                  </section>
                </main>
              </body>
            </html>
            """
        )

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> Response:
        if full_path.startswith("api/"):
            return JSONResponse(
                status_code=404,
                content={
                    "detail": "API endpoint not found",
                    "path": full_path,
                    "backend": "online",
                    "available_api_paths": [
                        "/api/health",
                        "/api/status",
                        "/api/runtime",
                        f"{settings.api_prefix}/analyze",
                        "/docs",
                    ],
                },
            )

        index_file = frontend_dist / "index.html"

        if index_file.exists():
            return FileResponse(str(index_file))

        return JSONResponse(
            status_code=404,
            content={
                "detail": "Frontend build not found. Run frontend with npm run dev or build it.",
                "backend": "online",
                "frontend_dev_url": settings.frontend_dev_url,
                "docs": "/docs",
            },
        )

    return app


app = create_app()