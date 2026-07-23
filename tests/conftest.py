"""pytest 启动时加载项目根目录的 .env（与 api/app.py 保持一致）。"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
