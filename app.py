from __future__ import annotations

import os

from server.app import app


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    import uvicorn

    resolved_port = port or int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host=host, port=resolved_port)


def uv_main():
    return app


if __name__ == "__main__":
    main()
