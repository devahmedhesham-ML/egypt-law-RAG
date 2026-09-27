# Egypt Law RAG

Arabic/English question answering over the Egyptian Civil Code, with answers cited by article number.
(ITI × MLOps MENA, Final Project 2: LLM / RAG.)

## Setup (WSL Ubuntu + CUDA)

Three virtual environments, each built from a lock file, because each one pins a different torch version:

| venv | lock file | used for |
|---|---|---|
| `~/venvs/egypt-law-rag` | `requirements.lock` | app: ingestion, embeddings, Chroma, RAGAS, BentoML, MLflow, DVC |
| `~/venvs/egypt-law-rag-serve` | `requirements-serve.lock` | `vllm serve` (OpenAI-compatible API) |
| `~/venvs/egypt-law-rag-quantize` | `requirements-quantize.lock` | one-off AWQ-4bit quantization |

```bash
sudo apt install -y build-essential   # C compiler: vLLM/Triton compile GPU kernels at start-up
curl -LsSf https://astral.sh/uv/install.sh | sh
for n in "" -serve -quantize; do
  uv venv --python 3.12 ~/venvs/egypt-law-rag$n
  VIRTUAL_ENV=~/venvs/egypt-law-rag$n uv pip sync requirements$n.lock
done
cp .env.example .env   # fill in keys
source ~/venvs/egypt-law-rag/bin/activate
dvc pull               # fetch the source PDF (public bucket, no AWS account needed)
```

The `requirements*.txt` files state intent. After editing one, regenerate its lock file:

```bash
uv pip compile --python-version 3.12 --python-platform x86_64-manylinux_2_28 requirements.txt -o requirements.lock
```

## LLM inference

One OpenAI-compatible client ([src/rag/llm/](src/rag/llm/)) drives two backends, chosen with `llm.backend` in [params.yaml](params.yaml):

| backend | server | model |
|---|---|---|
| `bedrock` | Amazon Bedrock's OpenAI-compatible endpoint (`OPENAI_BASE_URL`, key `Bedrock_API_key`) | `openai.gpt-oss-120b` |
| `vllm` | local `vllm serve` (`VLLM_BASE_URL`) | `Qwen/Qwen2.5-7B-Instruct-AWQ` |

The model answers only from the retrieved articles and cites them inline as `[Article 492]`. Every answer is checked: any cited article that was not retrieved is flagged as a hallucination.

```bash
# vLLM (serve venv); leave GPU memory for the embedding model
~/venvs/egypt-law-rag-serve/bin/vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --gpu-memory-utilization 0.75 --max-model-len 8192

# app venv
pytest                                        # unit tests, no GPU/AWS needed
python scripts/llm_smoke.py --backend bedrock # real call: 3 questions, streamed, citations checked
python scripts/llm_smoke.py --backend vllm
```

## Data

`data/raw/egyptian_civil_code.pdf` is tracked with DVC: git stores only the `.dvc` pointer file, and the PDF itself lives in S3 (`s3://amzn-egypt-law-rag/dvc`, eu-north-1).

The `dvc/` prefix of the bucket is publicly readable, so anyone can `dvc pull` without an AWS account. Only the owner can write (`dvc push`), using `aws login`: install AWS CLI v2 inside WSL and share one login with Windows:

```bash
ln -sfn /mnt/c/Users/<you>/.aws ~/.aws   # reuse the Windows ~/.aws
aws login
dvc push
```
