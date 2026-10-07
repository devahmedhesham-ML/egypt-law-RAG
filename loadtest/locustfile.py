"""Load test for the BentoML service (rubric R07): 50 concurrent users asking Civil Code questions.

    locust -f loadtest/locustfile.py --host http://localhost:3000 --headless -u 50 -r 5 -t 3m \
        --html reports/locust_report.html --csv reports/locust

Each simulated user asks a question from the evaluation set (Arabic and English), waits 1–3 s, and asks
again. 80% of requests use POST /ask (full answer), 20% POST /ask_stream (streamed), recorded twice: when the
first chunk of text arrives ("ask_stream: first chunk") and when the whole answer has ("ask_stream: whole answer").
"""

import json
import random
import time
from pathlib import Path

import requests
from locust import HttpUser, between, events, task

QUESTIONS = [json.loads(line)["question"]
             for line in (Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl").read_text(
                 encoding="utf-8").splitlines() if line.strip()]


class Asker(HttpUser):
    wait_time = between(1, 3)

    def on_start(self) -> None:
        self.raw = requests.Session()  # streams: Locust's own session reads the whole body before returning

    @task(4)
    def ask(self) -> None:
        with self.client.post("/ask", json={"question": random.choice(QUESTIONS)}, name="ask",
                              catch_response=True, timeout=300) as res:
            if res.status_code != 200:
                res.failure(f"HTTP {res.status_code}: {res.text[:120]}")
            elif "answer" not in res.json():
                res.failure("no answer in the response")

    @task(1)
    def ask_stream(self) -> None:
        # Timed by hand with a plain session: Locust's session reads the whole body first, which would make the
        # first chunk look as late as the last one.
        started = time.perf_counter()
        first, body, error = None, [], None
        try:
            with self.raw.post(f"{self.host}/ask_stream", json={"question": random.choice(QUESTIONS)},
                               stream=True, timeout=300) as res:
                for chunk in res.iter_content(chunk_size=64):
                    if first is None:
                        first = time.perf_counter() - started
                    body.append(chunk)
                text = b"".join(body).decode("utf-8", errors="replace")
                if res.status_code != 200 or "[error]" in text:
                    error = RuntimeError(f"HTTP {res.status_code}: {text[-120:]}")
        except Exception as e:  # noqa: BLE001 - recorded as a failed request
            error = e
        total = (time.perf_counter() - started) * 1000
        events.request.fire(request_type="POST", name="ask_stream: whole answer", response_time=total,
                            response_length=sum(map(len, body)), exception=error, context={})
        if first is not None:
            events.request.fire(request_type="POST", name="ask_stream: first chunk", response_time=first * 1000,
                                response_length=0, exception=None, context={})
