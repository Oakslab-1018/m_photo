"""Run with the independent inference interpreter, never the business venv."""
import argparse
import uvicorn
from .config import load_settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    settings = load_settings()
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not settings.api_key:
        parser.error("Non-loopback binding requires STREETCLIP_API_KEY")
    uvicorn.run("inference.streetclip.app:app", host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
