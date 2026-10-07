# Load test: 50 concurrent users (Locust)

**p95 latency at 50 concurrent users: 12 s for a full answer (`/ask`), 8.7 s until the first streamed text,
0 failures in 899 requests.** Full report: [locust_report.html](locust_report.html) (stats in `locust_stats.csv`).

## Setup

| | |
|---|---|
| Service | BentoML (`rag.serving.service:RagService`, async, 1 worker) → vLLM 0.30 serving **Qwen/Qwen2.5-7B-Instruct-AWQ** |
| Retrieval | Qwen3-Embedding-0.6B on the GPU, Chroma index, top 5 whole articles per question |
| Hardware | NVIDIA RTX 4070 Ti SUPER 16 GB (shared with the Windows desktop), AMD Ryzen 7 9700X, WSL2 |
| vLLM | `--gpu-memory-utilization 0.60 --max-model-len 8192` (`scripts/serve_vllm.sh`), greedy decoding |
| Load | 50 users, 5 new users/s, 3 minutes; each user asks a random question from `eval/questions.jsonl` (Arabic and English), waits 1–3 s, repeats |
| Mix | 80% `POST /ask` (full answer), 20% `POST /ask_stream` (timed at the first text chunk and at the whole answer) |
| Command | `locust -f loadtest/locustfile.py --host http://localhost:3000 --headless -u 50 -r 5 -t 3m --html reports/locust_report.html --csv reports/locust` |

## Results

| Request | Count | Failures | Median | p95 | p99 | Max |
|---|---|---|---|---|---|---|
| `/ask` (full answer) | 651 | 0 | 9.0 s | **12 s** | 16 s | 17.0 s |
| `/ask_stream`: first text | 124 | 0 | 7.4 s | **8.7 s** | 9.0 s | 9.1 s |
| `/ask_stream`: whole answer | 124 | 0 | 9.1 s | 11 s | 12 s | 12.6 s |
| All | 899 | 0 | 8.8 s | 12 s | 15 s | 17.0 s |

Throughput: **5.0 requests/s**. Without load, one question takes **0.9 s** end to end, and a streamed answer shows its
first text after **0.06–0.5 s**.

## Where the time goes

During the run vLLM generated **~460 tokens/s in total**, ran 6–12 requests at a time with **none waiting**, and used
under 16% of its KV cache: it is not short of memory, it is short of generation speed. Two checks confirm that the GPU's
generation throughput, not the API, sets the latency:

| 40 questions at once | Inside vLLM at once | Median latency | First streamed text |
|---|---|---|---|
| through FastAPI (`/ask`, all admitted) | 40 | 10.2 s | 7.6 s |
| through BentoML (`/ask`) | up to 22 | 9.0 s | 6.7 s |

Letting every request into vLLM does not make anyone faster: the same ~460 tokens/s are shared by more answers. And
moving the question-embedding model from the CPU to the GPU changed nothing measurable: the first run, with vLLM at
0.75 of the GPU (embedding on the CPU), gave `/ask` p95 **11 s** and p99 13 s over 614 requests, 0 failures
([locust_before_report.html](locust_before_report.html); its stream first-chunk timing was a Locust measuring bug, fixed
since). vLLM now runs at 0.60 anyway, which leaves the GPU room for the embedding model (15 ms per question instead of
56 ms on the CPU) with KV cache to spare.

## What would lower it

- **More generation throughput**: a second vLLM replica or a bigger GPU (the latency scales with the tokens shared).
- **Shorter answers**: `max_tokens` (1024 today) caps the long tail; most answers are far shorter.
- **The optimization module**: our own AWQ-4bit build and speculative decoding, measured with this same test.
- **Streaming in the client**: without queueing, users see text within half a second; under load the first text
  arrives ~2 s before the full answer.
