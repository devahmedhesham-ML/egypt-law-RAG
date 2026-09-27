"""Run the test console: python -m rag.ui [--port 7860]"""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Egypt Law RAG test console")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 only on a trusted network")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--reload", action="store_true", help="restart on code changes (development)")
    args = parser.parse_args()
    print(f"Test console: http://localhost:{args.port}")
    uvicorn.run("rag.ui.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
