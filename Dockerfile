# syntax=docker/dockerfile:1

FROM python:3.12-slim AS checkpoint-provider

ARG CHECKPOINT_URL=https://www.alessandroberti.it/checkpoint_rosetta_latest.tar.gz
ARG CHECKPOINT_SHA256=""

COPY docker/prepare_checkpoints.py /usr/local/bin/prepare_checkpoints.py
COPY . /context

RUN python /usr/local/bin/prepare_checkpoints.py \
        --local /context/checkpoints \
        --output /resolved-checkpoints \
        --url "${CHECKPOINT_URL}" \
        --sha256 "${CHECKPOINT_SHA256}"

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/proc-rosetta \
    PROC_ROSETTA_CHECKPOINT_DIR=/checkpoints \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50

RUN apt-get update \
    && apt-get install --yes --no-install-recommends graphviz libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system proc-rosetta \
    && useradd --system --gid proc-rosetta --create-home --home-dir /home/proc-rosetta proc-rosetta \
    && mkdir --parents /app /checkpoints \
    && chown proc-rosetta:proc-rosetta /app /checkpoints

WORKDIR /app

COPY --chown=proc-rosetta:proc-rosetta pyproject.toml README.md ./
COPY --chown=proc-rosetta:proc-rosetta src ./src
COPY --chown=proc-rosetta:proc-rosetta proc_rosetta_ui ./proc_rosetta_ui
COPY --chown=proc-rosetta:proc-rosetta pages ./pages
COPY --chown=proc-rosetta:proc-rosetta scripts/files ./scripts/files
COPY --chown=proc-rosetta:proc-rosetta .streamlit ./.streamlit
COPY --chown=proc-rosetta:proc-rosetta streamlit_app.py ./streamlit_app.py

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN python -m pip install --index-url "${PYTORCH_INDEX_URL}" "torch>=2.0" \
    && python -m pip install .

COPY --from=checkpoint-provider --chown=proc-rosetta:proc-rosetta \
    /resolved-checkpoints/ /checkpoints/

USER proc-rosetta

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"]

CMD ["streamlit", "run", "streamlit_app.py"]
