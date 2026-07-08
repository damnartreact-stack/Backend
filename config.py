from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp",
    "bmp",
    "tif",
    "tiff",
    "dxf",
    "dwg",
}

DEFAULT_CORS_ORIGINS = [
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


class Settings(BaseSettings):
    app_name: str = "FireDesign Automation API"
    environment: Literal["development", "production", "test"] = "development"

    api_prefix: str = "/api"
    host: str = "127.0.0.1"
    port: int = Field(default=8020, ge=1, le=65535)

    frontend_dev_url: str = "http://127.0.0.1:5174"

    max_upload_mb: int = Field(default=100, ge=1, le=500)

    allowed_extensions: set[str] = DEFAULT_ALLOWED_EXTENSIONS

    cors_origins: list[str] = DEFAULT_CORS_ORIGINS

    # Accuracy / OCR settings
    enable_ocr: bool = True
    enable_known_test_templates: bool = True
    enable_auto_scale: bool = True
    default_metres_per_pixel: float = Field(default=0.01, gt=0)
    default_min_room_area: float = Field(default=4.0, gt=0)
    default_socket_spacing: float = Field(default=4.0, gt=0)

    # Safety limits for image processing
    max_image_width_px: int = Field(default=5000, ge=500, le=12000)
    max_image_height_px: int = Field(default=5000, ge=500, le=12000)
    dxf_render_size_px: int = Field(default=2000, ge=800, le=5000)

    # FireDesign / Ceasefire POC features
    enable_ceasefire_bom: bool = True
    enable_updated_dxf_export: bool = True
    enable_export_package_zip: bool = True
    enable_review_gates: bool = True
    enable_dummy_prices: bool = True

    # Production limitation flags
    enable_direct_dwg_parsing: bool = False
    enable_native_hydraulic_solver: bool = False
    enable_ahj_submission_mode: bool = False

    # Output naming
    default_currency: str = "INR"
    output_package_name: str = "firedesign_ceasefire_export_package"
    updated_dxf_filename: str = "ceasefire_updated_layout.dxf"
    ceasefire_bom_filename: str = "ceasefire_bom.csv"

    # Optional Windows Tesseract path.
    # Example in .env:
    # FIRE_ALARM_TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
    tesseract_cmd: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FIRE_ALARM_",
        extra="ignore",
    )

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def normalize_allowed_extensions(cls, value: Any) -> set[str]:
        if value is None:
            return DEFAULT_ALLOWED_EXTENSIONS

        if isinstance(value, str):
            extensions = {
                item.strip().lower().lstrip(".")
                for item in value.split(",")
                if item.strip()
            }
            return extensions or DEFAULT_ALLOWED_EXTENSIONS

        if isinstance(value, (list, tuple, set)):
            extensions = {
                str(item).strip().lower().lstrip(".")
                for item in value
                if str(item).strip()
            }
            return extensions or DEFAULT_ALLOWED_EXTENSIONS

        return DEFAULT_ALLOWED_EXTENSIONS

    @field_validator("cors_origins", mode="before")
    @classmethod
    def normalize_cors_origins(cls, value: Any) -> list[str]:
        if value is None:
            return DEFAULT_CORS_ORIGINS

        if isinstance(value, str):
            origins = [
                item.strip().rstrip("/")
                for item in value.split(",")
                if item.strip()
            ]

            return list(dict.fromkeys(DEFAULT_CORS_ORIGINS + origins))

        if isinstance(value, (list, tuple, set)):
            origins = [
                str(item).strip().rstrip("/")
                for item in value
                if str(item).strip()
            ]

            return list(dict.fromkeys(DEFAULT_CORS_ORIGINS + origins))

        return DEFAULT_CORS_ORIGINS

    @field_validator("frontend_dev_url", mode="before")
    @classmethod
    def normalize_frontend_dev_url(cls, value: Any) -> str:
        value = str(value or "http://127.0.0.1:5174").strip()
        return value.rstrip("/")

    @field_validator("api_prefix", mode="before")
    @classmethod
    def normalize_api_prefix(cls, value: Any) -> str:
        text = str(value or "/api").strip()

        if not text.startswith("/"):
            text = f"/{text}"

        return text.rstrip("/") or "/api"

    @field_validator("default_currency", mode="before")
    @classmethod
    def normalize_currency(cls, value: Any) -> str:
        return str(value or "INR").strip().upper()

    @property
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parents[1]

    @property
    def backend_dir(self) -> Path:
        return self.base_dir / "backend"

    @property
    def frontend_dir(self) -> Path:
        return self.base_dir / "frontend"

    @property
    def frontend_dist(self) -> Path:
        return self.frontend_dir / "dist"

    @property
    def static_dir(self) -> Path:
        return self.frontend_dist / "assets"

    @property
    def samples_dir(self) -> Path:
        return self.base_dir / "samples"

    @property
    def samples_input_dir(self) -> Path:
        return self.samples_dir / "inputs"

    @property
    def samples_output_dir(self) -> Path:
        return self.samples_dir / "outputs"

    @property
    def docs_dir(self) -> Path:
        return self.base_dir / "docs"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def accepted_files_display(self) -> list[str]:
        return sorted(self.allowed_extensions)

    @property
    def capability_flags(self) -> dict[str, bool]:
        return {
            "ceasefire_bom_mapping": self.enable_ceasefire_bom,
            "updated_dxf_export": self.enable_updated_dxf_export,
            "export_package_zip": self.enable_export_package_zip,
            "review_gates": self.enable_review_gates,
            "dummy_prices": self.enable_dummy_prices,
            "dwg_direct_parsing": self.enable_direct_dwg_parsing,
            "native_hydraulic_solver": self.enable_native_hydraulic_solver,
            "ahj_submission_mode": self.enable_ahj_submission_mode,
        }

    @property
    def upload_help(self) -> dict[str, Any]:
        return {
            "max_upload_mb": self.max_upload_mb,
            "allowed_extensions": sorted(self.allowed_extensions),
            "best_accuracy": [
                "Use DXF with clean WALL, ROOM, DOOR and MEP layers.",
                "For PNG/JPG, use high resolution with readable room names and area labels.",
                "Avoid blurry screenshots or heavily compressed images.",
                "Convert DWG to DXF before analysis.",
            ],
            "outputs": [
                "Annotated PNG preview",
                "SVG preview",
                "Engineering report TXT",
                "Full JSON package",
                "Standard BOM CSV",
                "Calculation schedule CSV",
                "Ceasefire mapped BOM CSV",
                "Updated DXF drawing",
                "Export ZIP package",
            ],
        }


def _configure_tesseract(settings: Settings) -> None:
    """
    Configure pytesseract only if:
    1. OCR is enabled
    2. FIRE_ALARM_TESSERACT_CMD is provided
    3. pytesseract package is installed

    This keeps the backend alive even if OCR is not installed.
    """
    if not settings.enable_ocr:
        return

    if not settings.tesseract_cmd:
        return

    try:
        pytesseract_module = import_module("pytesseract")
        pytesseract_module.pytesseract.tesseract_cmd = settings.tesseract_cmd
    except Exception:
        pass


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    _configure_tesseract(settings)
    return settings