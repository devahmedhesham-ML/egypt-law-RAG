"""Embedding cost and speed for the chunking experiments: hardware, GPU utilization, index size, latency.

All GPU numbers come from NVML and are device-wide: they include anything else on the GPU (under WSL, the
Windows desktop too), so they are an upper bound for the embedding job itself.
"""

from __future__ import annotations

import os
import platform
import threading
import time
from pathlib import Path

import numpy as np

MB = 1024 * 1024


def hardware_info() -> dict[str, str]:
    """GPU, driver, CUDA, torch, CPU and RAM, as MLflow tags (strings)."""
    info = {"hw_os": platform.platform(), "hw_cpu_threads": str(os.cpu_count())}
    try:
        import torch  # version strings only: no CUDA context, which would hold GPU memory in this process

        info["hw_torch"] = torch.__version__
        info["hw_cuda"] = torch.version.cuda or "none"
    except Exception:  # noqa: BLE001
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name, driver = pynvml.nvmlDeviceGetName(handle), pynvml.nvmlSystemGetDriverVersion()
        info["hw_gpu"] = name.decode() if isinstance(name, bytes) else name
        info["hw_driver"] = driver.decode() if isinstance(driver, bytes) else driver
        info["hw_gpu_vram_gb"] = f"{pynvml.nvmlDeviceGetMemoryInfo(handle).total / 2**30:.1f}"
    except Exception:  # noqa: BLE001
        pass
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        info["hw_cpu"] = next(line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
                              if line.startswith("model name"))
        mem_kb = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                      if line.startswith("MemTotal"))
        info["hw_ram_gb"] = f"{mem_kb / 2**20:.1f}"
    except Exception:  # noqa: BLE001
        info.setdefault("hw_cpu", platform.processor() or "unknown")
    return info


class GpuSampler:
    """Samples device-wide GPU utilization (%) and memory used (MB) every `interval` seconds while active."""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.samples: list[tuple[float, int, int]] = []  # (time, util %, memory used MB)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.ended_at = 0.0

    def __enter__(self) -> GpuSampler:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml = pynvml
        except Exception:  # noqa: BLE001 - no NVIDIA GPU: no samples
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
                used = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used // MB
                self.samples.append((time.perf_counter(), int(util), int(used)))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval)

    def __exit__(self, *exc) -> None:
        self.ended_at = time.perf_counter()
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def idle(self, seconds: float = 1.0) -> None:
        """Sample for `seconds` before the job starts: what else is using the GPU (Windows desktop, other jobs)."""
        time.sleep(seconds)
        self.idle_until = time.perf_counter()

    def baseline(self) -> dict[str, float]:
        window = [s for s in self.samples if s[0] <= getattr(self, "idle_until", 0.0)]
        if not window:
            return {}
        return {"gpu_util_idle": float(np.mean([u for _, u, _ in window])),
                "vram_idle_mb": float(np.mean([m for _, _, m in window]))}

    def last(self, seconds: float) -> dict[str, float]:
        """Mean/max utilization and peak memory over the final `seconds` before the sampler stopped.

        The embedder loads its model first and embeds last, so its embedding phase is the final `embed_s`
        seconds of the sampled window; this leaves model loading out of the utilization numbers.
        """
        window = [s for s in self.samples if s[0] >= self.ended_at - seconds]
        if not window:
            return {}
        utils = [u for _, u, _ in window]
        return {"gpu_util_mean": float(np.mean(utils)), "gpu_util_max": float(max(utils)),
                "vram_peak_device_mb": float(max(m for _, _, m in window)), "gpu_util_samples": float(len(window)),
                **self.baseline()}


def dir_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / MB


def latency_stats(prefix: str, seconds: list[float]) -> dict[str, float]:
    ms = np.asarray(seconds) * 1000
    return {f"{prefix}_ms_p50": float(np.percentile(ms, 50)), f"{prefix}_ms_p95": float(np.percentile(ms, 95)),
            f"{prefix}_ms_mean": float(ms.mean())}


def query_latency(model_name: str, queries: list[str], device: str, max_seq_len: int, warmup: int = 3) -> dict:
    """Embed each question on its own, as production does, and time it (model already loaded, after warm-up).

    `cuda` uses fp16 like the production Retriever; `cpu` uses fp32 like the Docker image. Runs in a fresh
    subprocess, so its CUDA context and memory are gone before the next run's idle baseline is sampled.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        return pool.submit(_query_latency, model_name, queries, device, max_seq_len, warmup).result()


def _query_latency(model_name: str, queries: list[str], device: str, max_seq_len: int, warmup: int) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer

    from rag.ingest.embed import query_prompt_name

    if device == "cuda" and not torch.cuda.is_available():
        return {}
    kwargs = {"model_kwargs": {"dtype": torch.float16}} if device == "cuda" else {}
    started = time.perf_counter()
    model = SentenceTransformer(model_name, device=device, **kwargs)
    model.max_seq_length = max_seq_len
    load_s = time.perf_counter() - started
    prompt = query_prompt_name(model)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for q in queries[:warmup]:
        model.encode([q], prompt_name=prompt, normalize_embeddings=True)
    times = []
    with torch.inference_mode():
        for q in queries:
            t = time.perf_counter()
            model.encode([q], prompt_name=prompt, normalize_embeddings=True)  # returns numpy: waits for the GPU
            times.append(time.perf_counter() - t)
    tag = f"query_{device.replace('cuda', 'gpu')}"
    out = {**latency_stats(tag, times), f"{tag}_load_s": load_s}
    if device == "cuda":
        out[f"{tag}_vram_mb"] = torch.cuda.max_memory_allocated() / MB
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def search_latency(index_dir: Path, name: str, query_vectors: np.ndarray, k: int) -> dict:
    """Time one Chroma search per question against the run's index (after one warm-up search)."""
    from rag.ingest.store import search

    search(index_dir, name, query_vectors[:1], k=k)
    times = []
    for v in query_vectors:
        t = time.perf_counter()
        search(index_dir, name, v[None, :], k=k)
        times.append(time.perf_counter() - t)
    return latency_stats("search", times)
