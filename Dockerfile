FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# dependências de sistema (swig é exigido por algumas libs do FinRL)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential swig git \
    && rm -rf /var/lib/apt/lists/*

# instala dependências primeiro, aproveitando cache de camadas
COPY pyproject.toml poetry.lock* setup.py setup.cfg README.md ./
COPY finrl ./finrl
RUN pip install --upgrade pip && pip install -e .

# depois, o resto do código (scripts, config, etc.)
COPY . .

# por padrão, roda o orquestrador do experimento
CMD ["bash", "scripts/run_experiment.sh"]