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

## Data

`data/raw/egyptian_civil_code.pdf` is tracked with DVC: git stores only the `.dvc` pointer file, and the PDF itself lives in S3 (`s3://amzn-egypt-law-rag/dvc`, eu-north-1).

The `dvc/` prefix of the bucket is publicly readable, so anyone can `dvc pull` without an AWS account. Only the owner can write (`dvc push`), using `aws login`: install AWS CLI v2 inside WSL and share one login with Windows:

```bash
ln -sfn /mnt/c/Users/<you>/.aws ~/.aws   # reuse the Windows ~/.aws
aws login
dvc push
```
