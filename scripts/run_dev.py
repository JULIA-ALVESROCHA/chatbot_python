# scripts/run_dev.py
import sys
from pathlib import Path

# adiciona a raiz do projeto ao PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        "src.app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
