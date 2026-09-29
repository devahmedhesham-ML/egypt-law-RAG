"""GPU embedding with as many model replicas as fit safely, and live progress.

How the replica count is chosen:
1. Read free VRAM for the whole device (NVML), keep `vram_margin_gb` untouched.
2. Start one replica and give it the longest batch (worst case); the device-wide memory
   increase is what one replica really costs (CUDA context + weights + activations).
3. Start as many more as fit in the remaining budget, up to `max_replicas`. Each replica is a
   separate process capped at ~1.3x its measured footprint, and all pull from one job queue.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from tqdm import tqdm

MB = 1024 * 1024


def query_prompt_name(model) -> str | None:
    """Qwen3 embeds questions with its own "query" instruction; models without one (bge-m3) take them as-is."""
    return "query" if "query" in (getattr(model, "prompts", None) or {}) else None


@dataclass(frozen=True)
class GpuMemory:
    name: str
    total_mb: int
    used_mb: int  # whole device: every process, Windows too under WSL
    free_mb: int


_NVML_HANDLE = None


def gpu_memory(index: int = 0) -> GpuMemory | None:
    """Device-wide memory via NVML (fast), else nvidia-smi, else None (no NVIDIA GPU)."""
    global _NVML_HANDLE
    try:
        import pynvml

        if _NVML_HANDLE is None:
            pynvml.nvmlInit()
            _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
        name = pynvml.nvmlDeviceGetName(_NVML_HANDLE)
        name = name.decode() if isinstance(name, bytes) else name
        return GpuMemory(name, mem.total // MB, mem.used // MB, mem.free // MB)
    except Exception:  # noqa: BLE001 - fall through to nvidia-smi
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        name, total, used, free = (s.strip() for s in out.split(","))
        return GpuMemory(name, int(total), int(used), int(free))
    except Exception:  # noqa: BLE001
        return None


def gpu_utilization() -> int | None:
    """Percent of time the GPU's cores were busy over NVML's last sample window, or None if unavailable."""
    try:
        import pynvml

        if _NVML_HANDLE is None:
            gpu_memory()
        return int(pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE).gpu)
    except Exception:  # noqa: BLE001
        return None


def make_batches(token_counts: list[int], tokens_per_batch: int) -> list[list[int]]:
    """Length-sorted batches (longest first) whose padded size, items × longest, fits the budget."""
    order = sorted(range(len(token_counts)), key=lambda i: -token_counts[i])
    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for i in order:
        t = max(token_counts[i], 1)
        if current and (len(current) + 1) * max(longest, t) > tokens_per_batch:
            batches.append(current)
            current, longest = [], 0
        current.append(i)
        longest = max(longest, t)
    if current:
        batches.append(current)
    return batches


def plan_replicas(free_mb: float, margin_mb: float, per_replica_mb: float, max_replicas: int, n_batches: int) -> int:
    """How many replicas fit in free VRAM minus the margin (at least 1, at most one per batch)."""
    fit = int((free_mb - margin_mb) // per_replica_mb) if per_replica_mb > 0 else 1
    return max(1, min(fit, max_replicas, max(n_batches, 1)))


def weights_mb(model: str) -> float:
    """Size of the cached safetensors weights, or a conservative default."""
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(model, "model.safetensors")
        if isinstance(path, str) and os.path.exists(path):
            return os.path.getsize(path) / MB
    except Exception:  # noqa: BLE001
        pass
    return 1500.0


@dataclass(frozen=True)
class WorkerConfig:
    model: str
    max_seq_len: int
    mem_fraction: float | None
    offline: bool
    log_path: str


def _worker(rank: int, cfg: WorkerConfig, jobs, results) -> None:
    """One replica: load the model on the GPU in fp16, then embed batches until told to stop."""
    log = open(cfg.log_path, "w", buffering=1, encoding="utf-8")  # noqa: SIM115 - lives as long as the process
    os.dup2(log.fileno(), 1)  # compiler/library chatter goes to the log, not the progress bar
    os.dup2(log.fileno(), 2)
    sys.stdout = sys.stderr = log
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if cfg.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        if cfg.mem_fraction:
            torch.cuda.set_per_process_memory_fraction(cfg.mem_fraction, 0)
        model = SentenceTransformer(cfg.model, device="cuda", model_kwargs={"dtype": torch.float16})
        model.max_seq_length = cfg.max_seq_len
        results.put(("ready", rank, None))
        while (job := jobs.get()) is not None:
            kind, batch_id, texts = job
            with torch.inference_mode():
                vectors = model.encode(texts, batch_size=len(texts),
                                       prompt_name=query_prompt_name(model) if kind == "query" else None,
                                       normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            results.put(("done", rank, (kind, batch_id, np.asarray(vectors, dtype=np.float32))))
        results.put(("exit", rank, None))
    except BaseException:  # noqa: BLE001 - report every failure (OOM included) to the parent
        results.put(("error", rank, traceback.format_exc()))


@dataclass
class EmbedStats:
    device: str
    gpu: str | None
    replicas: int
    per_replica_mb: float
    free_before_mb: int
    margin_mb: int
    peak_used_mb: int
    total_mb: int
    batches: int
    tokens_per_batch: int
    texts: int
    tokens: int
    load_s: float
    embed_s: float
    total_s: float

    @property
    def tokens_per_s(self) -> float:
        return self.tokens / self.embed_s if self.embed_s else 0.0

    @property
    def texts_per_s(self) -> float:
        return self.texts / self.embed_s if self.embed_s else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tokens_per_s"] = round(self.tokens_per_s, 1)
        d["texts_per_s"] = round(self.texts_per_s, 2)
        return d


class GpuEmbedder:
    def __init__(self, model: str, *, tokens_per_batch: int, max_replicas: int, margin_gb: float,
                 max_seq_len: int, replicas: int | None = None, force_cpu: bool = False,
                 log: Callable[[str], None] = print) -> None:
        self.model = model
        self.tokens_per_batch = tokens_per_batch
        self.max_replicas = max_replicas
        self.margin_mb = int(margin_gb * 1024)
        self.max_seq_len = max_seq_len
        self.replicas = replicas
        self.force_cpu = force_cpu
        self.log = log

    def run(self, docs: list[str], doc_tokens: list[int], queries: list[str] = ()) -> tuple[np.ndarray, np.ndarray, EmbedStats]:
        gpu = None if self.force_cpu else gpu_memory()
        if gpu is None:
            self.log("[yellow]! No NVIDIA GPU visible: embedding on CPU (slow).[/]")
            return self._run_cpu(docs, doc_tokens, list(queries))
        need = weights_mb(self.model) * 1.15 + 700
        if gpu.free_mb < need:
            self.log(f"[yellow]! Only {gpu.free_mb / 1024:.1f} GB VRAM free, one replica needs ~{need / 1024:.1f} GB: "
                     "embedding on CPU. Stop vLLM or other GPU jobs to use the GPU.[/]")
            return self._run_cpu(docs, doc_tokens, list(queries))
        if gpu.free_mb - self.margin_mb < need:
            self.log(f"[yellow]! {gpu.free_mb / 1024:.1f} GB free leaves less than the {self.margin_mb / 1024:.1f} GB "
                     "safety margin after one replica: running a single replica.[/]")
        return self._run_gpu(docs, doc_tokens, list(queries), gpu, need)

    # --- GPU ----------------------------------------------------------------------------
    def _run_gpu(self, docs, doc_tokens, queries, gpu: GpuMemory, need_mb: float):
        started = time.perf_counter()
        batches = make_batches(doc_tokens, self.tokens_per_batch)
        query_batches = [list(range(i, min(i + 32, len(queries)))) for i in range(0, len(queries), 32)]
        offline = weights_mb(self.model) != 1500.0
        log_dir = tempfile.gettempdir()
        ctx = mp.get_context("spawn")
        jobs, results = ctx.Queue(), ctx.Queue()
        procs: list = []
        doc_vecs: dict[int, np.ndarray] = {}
        query_vecs: dict[int, np.ndarray] = {}

        def start(rank: int, fraction: float | None) -> None:
            cfg = WorkerConfig(self.model, self.max_seq_len, fraction, offline,
                               os.path.join(log_dir, f"rag_embed_worker_{rank}.log"))
            p = ctx.Process(target=_worker, args=(rank, cfg, jobs, results), daemon=True)
            p.start()
            procs.append(p)

        def receive(timeout: float):
            try:
                msg = results.get(timeout=timeout)
            except queue.Empty:
                dead = [i for i, p in enumerate(procs) if not p.is_alive() and p.exitcode not in (0, None)]
                if dead:
                    raise RuntimeError(f"replica {dead[0]} died (exit code {procs[dead[0]].exitcode}); see "
                                       f"{os.path.join(log_dir, f'rag_embed_worker_{dead[0]}.log')}") from None
                return None
            if msg[0] == "error":
                raise RuntimeError(f"replica {msg[1]} failed:\n{msg[2]}")
            return msg

        try:
            # 1) probe replica: load, then embed the longest batch to measure the real footprint
            first_fraction = min(0.95, max(gpu.free_mb - self.margin_mb, need_mb) / gpu.total_mb)
            start(0, first_fraction)
            with tqdm(total=1, desc="Loading replica 1", bar_format="{desc}: {elapsed}", leave=False) as spin:
                while (msg := receive(1.0)) is None or msg[0] != "ready":
                    spin.refresh()
            load_s = time.perf_counter() - started
            embed_started = time.perf_counter()
            jobs.put(("doc", 0, [docs[i] for i in batches[0]]))
            while (msg := receive(1.0)) is None or msg[0] != "done":
                pass
            doc_vecs[0] = msg[2][2]
            after = gpu_memory() or gpu
            per_replica = max(after.used_mb - gpu.used_mb, need_mb * 0.8) * 1.10
            safe = plan_replicas(gpu.free_mb, self.margin_mb, per_replica, self.max_replicas, len(batches))
            done_early = {0}

            # 2) memory allows more replicas, but do they help? Let replica 1 work alone on the next
            #    batches and watch GPU utilization: if one replica already keeps the cores busy,
            #    more replicas only compete for them (measured on an RTX 4070 Ti SUPER: 1 replica is fastest).
            busy = None
            if safe > 1 and not self.replicas:
                # 2.5 s of work for replica 1 alone, two batches in flight so it never idles; NVML's
                # utilization lags ~0.5 s behind the load, so samples start after 0.7 s.
                samples: list[int] = []
                in_flight: set[int] = set()
                next_batch, warm_started = 1, time.perf_counter()
                while in_flight or (next_batch < len(batches) and time.perf_counter() - warm_started < 2.5):
                    while len(in_flight) < 2 and next_batch < len(batches) and time.perf_counter() - warm_started < 2.5:
                        jobs.put(("doc", next_batch, [docs[i] for i in batches[next_batch]]))
                        in_flight.add(next_batch)
                        done_early.add(next_batch)
                        next_batch += 1
                    msg = receive(0.1)
                    if msg and msg[0] == "done":
                        doc_vecs[msg[2][1]] = msg[2][2]
                        in_flight.discard(msg[2][1])
                    if time.perf_counter() - warm_started > 0.7 and (u := gpu_utilization()) is not None:
                        samples.append(u)
                busy = round(sum(samples) / len(samples)) if len(samples) >= 3 else None
            n = safe
            reason = f"{safe} fit in memory"
            if self.replicas:
                n = min(self.replicas, safe)
                reason = f"{self.replicas} requested" + (f", only {safe} fit safely" if self.replicas > safe else "")
            elif busy is not None and busy >= 90 and safe > 1:
                n = 1
                reason = f"{safe} would fit, but one replica already keeps the GPU {busy}% busy"
            self.log(f"GPU {gpu.name}: {gpu.free_mb / 1024:.1f} GB free of {gpu.total_mb / 1024:.1f} GB · one replica "
                     f"uses ~{per_replica / 1024:.2f} GB (measured) · margin {self.margin_mb / 1024:.1f} GB "
                     f"→ [bold]{n} replica{'s' if n != 1 else ''}[/] ({reason}) · {len(batches)} batches of "
                     f"≤{self.tokens_per_batch:,} padded tokens")

            # 3) the other replicas, each capped near the measured footprint
            fraction = min(0.95, per_replica * 1.3 / gpu.total_mb)
            for rank in range(1, n):
                start(rank, fraction)
            for b in range(1, len(batches)):
                if b not in done_early:
                    jobs.put(("doc", b, [docs[i] for i in batches[b]]))
            for q, idx in enumerate(query_batches):
                jobs.put(("query", q, [queries[i] for i in idx]))
            for _ in range(n):
                jobs.put(None)

            # 4) progress until every batch is back
            pending = ({("doc", b) for b in range(1, len(batches)) if b not in done_early}
                       | {("query", q) for q in range(len(query_batches))})
            tokens_done = sum(doc_tokens[i] for b in done_early for i in batches[b])
            ready, peak, last_poll = 1, after.used_mb, 0.0
            bar = tqdm(total=len(docs), initial=sum(len(batches[b]) for b in done_early), unit="chunk",
                       desc="Embedding", dynamic_ncols=True)
            while pending:
                msg = receive(0.5)
                if msg and msg[0] == "ready":
                    ready += 1
                elif msg and msg[0] == "done":
                    kind, batch_id, vectors = msg[2]
                    pending.discard((kind, batch_id))
                    if kind == "doc":
                        doc_vecs[batch_id] = vectors
                        bar.update(len(batches[batch_id]))
                        tokens_done += sum(doc_tokens[i] for i in batches[batch_id])
                    else:
                        query_vecs[batch_id] = vectors
                now = time.perf_counter()
                if now - last_poll > 0.5:
                    mem = gpu_memory()
                    if mem:
                        peak = max(peak, mem.used_mb)
                    bar.set_postfix_str(
                        f"{tokens_done / (now - embed_started):,.0f} tok/s · replicas {ready}/{n} · "
                        f"VRAM {(mem.used_mb if mem else 0) / 1024:.1f}/{gpu.total_mb / 1024:.1f} GB", refresh=True)
                    last_poll = now
            bar.close()
            embed_s = time.perf_counter() - embed_started
            for p in procs:
                p.join(timeout=30)
        finally:
            for p in procs:
                if p.is_alive():
                    p.terminate()

        stats = EmbedStats("cuda", gpu.name, n, round(per_replica), gpu.free_mb, self.margin_mb, peak, gpu.total_mb,
                           len(batches), self.tokens_per_batch, len(docs), sum(doc_tokens),
                           round(load_s, 2), round(embed_s, 2), round(time.perf_counter() - started, 2))
        return self._assemble(docs, batches, doc_vecs, query_batches, query_vecs, stats)

    # --- CPU fallback ------------------------------------------------------------------------
    def _run_cpu(self, docs, doc_tokens, queries):
        started = time.perf_counter()
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model, device="cpu")
        model.max_seq_length = self.max_seq_len
        load_s = time.perf_counter() - started
        batches = make_batches(doc_tokens, min(self.tokens_per_batch, 4096))
        doc_vecs: dict[int, np.ndarray] = {}
        embed_started = time.perf_counter()
        with tqdm(total=len(docs), unit="chunk", desc="Embedding (CPU)", dynamic_ncols=True) as bar:
            for b, idx in enumerate(batches):
                doc_vecs[b] = model.encode([docs[i] for i in idx], batch_size=len(idx), normalize_embeddings=True,
                                           convert_to_numpy=True, show_progress_bar=False)
                bar.update(len(idx))
        query_batches = [list(range(len(queries)))] if queries else []
        query_vecs = {0: model.encode(list(queries), prompt_name=query_prompt_name(model), normalize_embeddings=True,
                                      convert_to_numpy=True)} if queries else {}
        embed_s = time.perf_counter() - embed_started
        stats = EmbedStats("cpu", None, 1, 0, 0, 0, 0, 0, len(batches), min(self.tokens_per_batch, 4096), len(docs),
                           sum(doc_tokens), round(load_s, 2), round(embed_s, 2), round(time.perf_counter() - started, 2))
        return self._assemble(docs, batches, doc_vecs, query_batches, query_vecs, stats)

    @staticmethod
    def _assemble(docs, batches, doc_vecs, query_batches, query_vecs, stats):
        dim = next(iter(doc_vecs.values())).shape[1]
        out = np.zeros((len(docs), dim), dtype=np.float32)
        for b, idx in enumerate(batches):
            out[idx] = doc_vecs[b]
        queries_out = (np.concatenate([query_vecs[q] for q in range(len(query_batches))])
                       if query_batches else np.zeros((0, dim), dtype=np.float32))
        return out, queries_out, stats
