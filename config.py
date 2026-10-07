"""新 Geo 系统共享配置：AI 模式、只读数据目录与推理参数。

legacy Flask 的 SECRET_KEY、数据库 URI 与 PyInstaller 路径 helper 已随
旧应用归档到 .local-backup/legacy-2026-10-07/，不再保留在运行配置中。
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# mock 为默认值：镜像/本地无需 easyocr 模型即可启动。
AI_MODE = os.environ.get("GEO_AI_MODE", "mock")

# 推理引擎参数（backend/app/domain/geolocation 当前实际使用）。
MIN_CLUE_CONFIDENCE = 0.3
TOP_N_HYPOTHESES = 5
CONFIRMATION_THRESHOLD = 0.65