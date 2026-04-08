from __future__ import annotations

from fastapi import FastAPI


app = FastAPI(
    title="SupportOpsEnv Server",
    description="Minimal server entry point for OpenEnv validation and deployment hooks.",
    version="0.1.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "support-ops-env",
        "status": "ok",
        "message": "SupportOpsEnv server entry point is available.",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)

def uv_main():
	return app

if __name__ == "__main__":
    main()
