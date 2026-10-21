"""HTTP server with a Jev/basal-compatible ``POST /v1/systemone`` endpoint.

    uv run python -m bullfinch.server --port 8000 [--adapter artifacts/pilot/adapter]

Copied from Serin 4B (axiomjudge); keep the two in step by hand.
"""

import argparse
from pathlib import Path

# Module-level import on purpose: FastAPI resolves the `Request` annotation by name, which
# fails for a class imported inside build_app.
from fastapi import FastAPI, HTTPException, Request


def build_app(engine):
    app = FastAPI(title="Bullfinch")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "model": engine.model_name, "adapter": engine.adapter}

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> dict:
        body = await request.json()
        try:
            return engine.systemone(body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--adapter", help="LoRA adapter directory (default: unfine-tuned base)")
    ap.add_argument("--calibration", type=Path, help="CALIBRATION.json with per-type temperatures")
    ap.add_argument("--tta", action="store_true",
                    help="average every option rotation in both frame languages (2k passes per question)")
    args = ap.parse_args()

    import uvicorn

    from .engine import DecisionEngine

    engine = DecisionEngine(adapter=args.adapter, calibration=args.calibration, tta=args.tta)
    uvicorn.run(build_app(engine), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
