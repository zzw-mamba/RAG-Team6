# Build on a Linux Docker host; do not assume Docker-in-Docker is available on AutoDL.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
WORKDIR /opt/rag-team6/autodl
ENV PYTHONUNBUFFERED=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=20 PIP_RESUME_RETRIES=30
RUN python -m venv --system-site-packages .venv
ENV PATH="/opt/rag-team6/autodl/.venv/bin:$PATH"
COPY . .
RUN python -m pip install --upgrade 'pip>=25.2,<27' && python -m pip install -e '.[gpu]'
CMD ["bash", "run_all.sh", "--skip-install"]
