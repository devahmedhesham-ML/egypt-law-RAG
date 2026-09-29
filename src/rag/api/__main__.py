"""Run the Q&A API: python -m rag.api [--host 0.0.0.0] [--port 8000]"""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Egypt Law RAG API")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 inside a container")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("rag.api.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
