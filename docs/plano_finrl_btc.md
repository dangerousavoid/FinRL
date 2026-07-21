# Plano de Implantação — FinRL + Bitcoin (CryptoDataDownload)

Runbook passo a passo para transformar o projeto **AI4Finance-Foundation/FinRL** em algo concreto: rodando com seus próprios dados históricos de Bitcoin, com ambiente de dev na nuvem, CI/CD e deploy fácil de "subir" e "derrubar" numa VPS.

> **Filosofia do plano:** você **não vai refatorar** o FinRL. Tudo que adicionamos são **arquivos novos** (adaptador de dados, Docker, workflows de CI/CD) empilhados por cima. O único arquivo *existente* que você edita é o `finrl/config.py` — e apenas os **parâmetros hard-coded** (datas e lista de indicadores).

---

## 0. Panorama da arquitetura

| Necessidade sua | Solução escolhida | Por quê |
|---|---|---|
| Não tenho ambiente nem IDE | **GitHub Codespaces** (VS Code na nuvem) | Zero instalação local; roda agentes de IA; é GitHub puro |
| IDE com agentes de IA | VS Code (Codespaces ou Desktop) + Copilot / Claude | Suportado nativamente |
| CI/CD | **GitHub Actions** | Lint + testes + build da imagem + deploy automático |
| Deploy organizado, fácil subir/derrubar | **Docker Compose** na VPS | `docker compose up -d` / `down` |
| Máximo de GitHub | Fork + Codespaces + Actions + GHCR (Container Registry) | Tudo dentro do ecossistema |
| Dados de Bitcoin | CSV da CryptoDataDownload via **adaptador** | Não mexe no core do FinRL |
| VPS (se necessário) | **Hostinger KVM** | Alvo do deploy e de treinos longos |

**Fluxo geral:**

```
CryptoDataDownload (CSV BTC — 1 minuto, o mais granular gratuito)
        │  (adaptador cdd_to_finrl.py — arquivo novo; reamostra p/ 5m/15m/1h se quiser)
        ▼
train_data.csv / trade_data.csv   ← mesmo formato que o FinRL já consome
        │
        ▼
FinRL: treino (A2C, PPO, DDPG, TD3, SAC)  →  backtest
        │
        ▼
Imagem Docker  →  GHCR  →  VPS Hostinger (docker compose up -d)
```

**Sobre poder computacional (importante — dados de 1 minuto):** este projeto usa a granularidade **mais fina gratuita da CryptoDataDownload: 1 minuto**. Isso significa **~4 milhões de linhas** (BTC desde ~2017) e um CSV de **centenas de MB a ~1 GB**. A VPS da Hostinger é **CPU-only** (sem GPU), e treinar DRL passo-a-passo sobre milhões de timesteps por episódio pode levar **horas a dias por agente**. Estratégia recomendada:

1. **Desenvolva num subconjunto** (poucos meses de minuto) para iterar rápido; só depois rode o histórico completo.
2. **Decida conscientemente a granularidade de treino:** 1 minuto puro (máxima fidelidade, máximo custo) vs. **reamostrar** para 5min/15min/1h (reduz linhas em 5–60× mantendo o caráter intradiário). Reamostrar é **uma opção no adaptador** — não é refactor do FinRL. O plano abaixo já traz essa opção.
3. **Dimensione a VPS para memória**, não só CPU: ver Fase 6.

> **Nota:** o README do FinRL agora recomenda o sucessor **FinRL-X / FinRL-Trading** para produção. Como você pediu o FinRL clássico, o plano é sobre ele. Se depois quiser algo mais "production-grade", o FinRL-X é o caminho — mas o clássico é mais simples para reaquecer os conhecimentos.

---

## FASE 1 — Contas e fork (10 min)

**Pré-requisitos:** apenas uma conta no GitHub. Nada instalado localmente.

1. Crie/entre na sua conta GitHub.
2. Acesse `https://github.com/AI4Finance-Foundation/FinRL` e clique em **Fork** (canto superior direito). Isso cria `https://github.com/SEU_USUARIO/FinRL`.
   - Fork (e não `git clone` local) é o certo aqui: dá base para Actions, Codespaces e histórico próprio — maximiza o GitHub, como você pediu.
3. No seu fork, deixe o branch `master` como está e crie um branch de trabalho depois (Fase 3).

---

## FASE 2 — Ambiente de desenvolvimento com Codespaces (30 min)

A ideia: um contêiner de dev reproduzível descrito em código (`devcontainer`), que o Codespaces sobe na nuvem. Você abre o VS Code no navegador (ou conecta o VS Code Desktop) e tudo já vem pronto.

### 2.1 Criar o devcontainer

No seu fork, crie o arquivo `.devcontainer/devcontainer.json`:

```jsonc
{
  "name": "finrl-dev",
  "image": "mcr.microsoft.com/devcontainers/python:3.10-bookworm",
  "features": {
    "ghcr.io/devcontainers/features/git:1": {}
  },
  "postCreateCommand": "bash .devcontainer/setup.sh",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.black-formatter",
        "ms-toolsai.jupyter",
        "GitHub.copilot"
      ]
    }
  },
  "hostRequirements": { "cpus": 4, "memory": "8gb" }
}
```

E o `.devcontainer/setup.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Dependências de sistema úteis para o stable-baselines3[extra] / gymnasium
sudo apt-get update
sudo apt-get install -y --no-install-recommends build-essential swig

python -m pip install --upgrade pip

# IMPORTANTE: use o pyproject (pip install -e .), NÃO o requirements.txt.
# O requirements.txt inclui TA-Lib (exige lib de sistema e quebra a build);
# o pyproject não inclui e os indicadores usam stockstats.
pip install -e .

echo "Ambiente FinRL pronto."
```

> Por que Python 3.10? O `pyproject.toml` do FinRL aceita `^3.7` (até 3.12), mas 3.10 é o ponto mais estável com `stable-baselines3` 2.x + `ray` 2.x. Se algo reclamar, 3.11 também funciona.

### 2.2 Subir o Codespace

1. No seu fork, botão verde **Code → Codespaces → Create codespace on master**.
2. Aguarde ~3–5 min (ele executa o `setup.sh`).
3. Terminal do Codespace: valide a instalação com um teste rápido de import:
   ```bash
   python -c "import finrl; from finrl.meta.preprocessor.preprocessors import FeatureEngineer; print('OK')"
   ```
4. **(Opcional — smoke test)** rode o exemplo original de ações para confirmar que o pipeline funciona ponta-a-ponta:
   ```bash
   python examples/FinRL_StockTrading_2026_1_data.py   # baixa DOW30 do Yahoo, gera train/trade
   ```
   Se gerar `train_data.csv` e `trade_data.csv`, seu ambiente está 100%.

> **Custo/limites:** o plano gratuito do GitHub dá uma cota mensal de horas de Codespaces (2-core). Pare o Codespace quando não estiver usando (**Codespaces → Stop**) para não consumir cota. Ele é ephemeral: "sobe e derruba" à vontade.

---

## FASE 3 — Plugar os dados do Bitcoin (o coração do projeto) (1–2 h)

### 3.1 Baixar o CSV na CryptoDataDownload (granularidade de 1 minuto)

1. **Crie uma conta gratuita** em `https://www.cryptodatadownload.com/` — os arquivos **de hora e de minuto** exigem login grátis (o diário é aberto). Sem rate limit e sem paywall por tamanho.
2. Escolha uma exchange (ex.: **Binance**, par **BTC/USDT**) e a granularidade **Minute (1m)**.
3. Baixe o CSV. Formato:
   - **1ª linha** é um comentário (a URL do site) → precisa ser pulada na leitura (`skiprows=1`).
   - Colunas: `unix, date, symbol, open, high, low, close, Volume BTC, Volume USDT` (nome da coluna de volume varia). Em arquivos de minuto, a coluna `date` traz **timestamp completo** (`YYYY-MM-DD HH:MM:SS`).
   - Ordem: **decrescente** por data → precisa reordenar (ascendente).
4. Coloque em `data/raw/btcusdt_1min.csv`. (Crie a pasta `data/raw/`.)

> **Atenção ao tamanho:** o arquivo de minuto tem ~4M linhas e pode passar de **1 GB**. **Não versione no Git.** Adicione ao `.gitignore`:
> ```
> data/raw/*.csv
> !data/raw/sample_*.csv
> ```
> Versione apenas um **recorte pequeno** (`data/raw/sample_btcusdt_1min.csv`, ex.: 1 mês) para os testes de CI. O arquivo grande fica só na sua máquina/Codespace e é enviado à VPS via `scp` (Fase 6).

### 3.2 Criar o adaptador (arquivo novo — não é refactor)

Crie `scripts/cdd_to_finrl.py`:

```python
from __future__ import annotations
import argparse
import pandas as pd
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config


def load_cdd(csv_path: str, tic: str = "BTCUSDT", resample: str | None = None) -> pd.DataFrame:
    """Lê um CSV (1 minuto) da CryptoDataDownload e devolve o DataFrame no
    contrato que o FinRL espera: date, open, high, low, close, volume, tic.

    resample: None mantém 1 minuto. Ou passe uma regra pandas: '5min', '15min', '1h'.
    """
    # 1ª linha é comentário (URL) -> skiprows=1
    raw = pd.read_csv(csv_path, skiprows=1)
    raw.columns = [c.strip().lower() for c in raw.columns]

    # coluna de volume varia: 'volume btc', 'volume usdt', 'volume base'...
    vol_col = next(
        (c for c in raw.columns if c.startswith("volume") and ("btc" in c or "base" in c)),
        None,
    ) or next(c for c in raw.columns if c.startswith("volume"))
    raw["volume"] = raw[vol_col]

    # timestamp COMPLETO (intradiário) — não corte para só a data
    raw["dt"] = pd.to_datetime(raw["date"])
    raw = raw[["dt", "open", "high", "low", "close", "volume"]].copy()
    raw = raw.sort_values("dt").drop_duplicates("dt").set_index("dt")

    # reamostragem opcional para reduzir o nº de linhas (OHLCV correto)
    if resample:
        raw = raw.resample(resample).agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}
        ).dropna(subset=["open"])

    df = raw.reset_index()
    df["tic"] = tic
    df["date"] = df["dt"].dt.strftime("%Y-%m-%d %H:%M:%S")  # datetime como string
    return df[["date", "open", "high", "low", "close", "volume", "tic"]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="caminho do CSV (1 minuto) da CryptoDataDownload")
    p.add_argument("--tic", default="BTCUSDT")
    p.add_argument("--resample", default=None,
                   help="regra pandas p/ reamostrar: 5min, 15min, 1h. Vazio = 1 minuto puro")
    p.add_argument("--train-out", default="train_data.csv")
    p.add_argument("--trade-out", default="trade_data.csv")
    args = p.parse_args()

    df = load_cdd(args.csv, tic=args.tic, resample=args.resample)
    print(f"linhas após leitura/reamostragem: {len(df):,}")

    # FeatureEngineer com use_vix=False (VIX é do mercado de ações, não de cripto)
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(df)

    # bounds intradiários: aceita "2018-01-01" ou "2018-01-01 00:00:00"
    train = data_split(processed, config.TRAIN_START_DATE, config.TRAIN_END_DATE)
    trade = data_split(processed, config.TRADE_START_DATE, config.TRADE_END_DATE)

    train.to_csv(args.train_out, index=False)
    trade.to_csv(args.trade_out, index=False)
    print(f"train: {train.shape}  trade: {trade.shape}")
    print(f"período train: {train.date.min()} -> {train.date.max()}")
    print(f"período trade: {trade.date.min()} -> {trade.date.max()}")


if __name__ == "__main__":
    main()
```

**Por que isso respeita seu requisito:** este script gera `train_data.csv` e `trade_data.csv` no **mesmo formato** que o passo 1 do exemplo oficial gera. Os scripts de treino e backtest do FinRL consomem esses arquivos sem alteração. Você está **substituindo a fonte de dados**, não reescrevendo o FinRL.

### 3.3 Ajustar os parâmetros hard-coded (única edição em arquivo existente)

Abra `finrl/config.py` e ajuste as **datas** para casarem com o histórico do seu CSV. Com dados intradiários, use **timestamps completos** nos limites (mais explícito e seguro que só a data):

```python
# Exemplo — ajuste conforme o alcance do SEU csv (1 minuto, desde ~2017)
TRAIN_START_DATE = "2018-01-01 00:00:00"
TRAIN_END_DATE   = "2023-12-31 23:59:00"

TRADE_START_DATE = "2024-01-01 00:00:00"
TRADE_END_DATE   = "2025-06-30 23:59:00"
```

Sobre a lista `INDICATORS` (MACD, Bollinger, RSI, CCI, DX, SMAs): ela funciona, **mas as janelas são contadas em linhas, não em tempo de calendário**. Com barras de 1 minuto, `rsi_30`/`close_30_sma` significam "30 minutos", não "30 dias". Se quiser janelas mais longas equivalentes, aumente os números (ex.: `close_240_sma` = 4 horas em barras de 1 min). Editar essa lista é outro ajuste legítimo de parâmetro hard-coded.

Esses (datas e `INDICATORS`) são exatamente os "parâmetros hard-coded" que você mencionou — nenhuma lógica do FinRL é tocada.

### 3.4 Rodar o pipeline com dados de Bitcoin

**Comece pequeno para iterar rápido.** Rode primeiro num recorte ou reamostrado; só depois no minuto puro completo.

```bash
# OPÇÃO A (recomendada p/ desenvolver): reamostrar p/ 1h reduz ~60x as linhas
python scripts/cdd_to_finrl.py --csv data/raw/btcusdt_1min.csv --tic BTCUSDT --resample 1h

# OPÇÃO B: intradiário mais fino, custo maior
python scripts/cdd_to_finrl.py --csv data/raw/btcusdt_1min.csv --tic BTCUSDT --resample 5min

# OPÇÃO C: 1 minuto puro (máxima fidelidade, treino lento em CPU)
python scripts/cdd_to_finrl.py --csv data/raw/btcusdt_1min.csv --tic BTCUSDT

# 2) treinar os agentes (usa train_data.csv)
python examples/FinRL_StockTrading_2026_2_train.py

# 3) backtest (usa trade_data.csv, gera backtest_result.png)
python examples/FinRL_StockTrading_2026_3_Backtest.py
```

> **Ajuste `timesteps` de treino.** Os scripts de exemplo têm um número de `total_timesteps` hard-coded pensado para dados diários. Com milhões de barras de minuto, mantê-lo baixo evita treinos intermináveis — comece pequeno (ex.: 50k–100k), valide o pipeline, e só então aumente. Esse é mais um parâmetro hard-coded editável.
>
> Se algum script de exemplo tiver `tickers`/caminho hard-coded que conflite com cripto, edite o parâmetro (não a lógica). Comece com um ativo só (BTCUSDT).
>
> **Alternativa nativa:** o FinRL tem uma aplicação `finrl/applications/high_frequency_trading` e um env `env_cryptocurrency_trading/env_btc_ccxt.py`. Se quiser um ambiente pensado para alta frequência em vez de reutilizar o env de ações, vale explorar — mas exige mais leitura de código. Para o caminho "trocar dados sem refatorar", a abordagem acima (reutilizar o pipeline de exemplo) é a mais direta.

### 3.5 Commit

```bash
git checkout -b feat/btc-data
# NÃO adicione o CSV grande de minuto. Versione só um sample pequeno.
git add .devcontainer scripts data/raw/sample_btcusdt_1min.csv finrl/config.py .gitignore
git commit -m "feat: adaptador CryptoDataDownload (1min) + reamostragem + datas BTC + devcontainer"
git push -u origin feat/btc-data
```

Abra um Pull Request no seu fork (bom hábito e ativa o CI da próxima fase).

---

## FASE 4 — Dockerizar para deploy reproduzível (1 h)

Objetivo: uma imagem que roda o pipeline (ou um Jupyter/serviço) e um `docker-compose.yml` para subir/derrubar num comando.

### 4.1 Dockerfile (crie na raiz: `Dockerfile`)

```dockerfile
FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential swig git \
    && rm -rf /var/lib/apt/lists/*

# instala dependências primeiro (cache de camadas)
COPY pyproject.toml poetry.lock* setup.py setup.cfg ./
COPY finrl ./finrl
RUN pip install --upgrade pip && pip install -e .

# resto do código
COPY . .

# roda o pipeline completo por padrão
CMD ["bash", "scripts/run_pipeline.sh"]
```

### 4.2 Script de pipeline (crie `scripts/run_pipeline.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

CSV="${CSV_PATH:-data/raw/btcusdt_1min.csv}"
RESAMPLE_ARG=""
[ -n "${RESAMPLE:-}" ] && RESAMPLE_ARG="--resample ${RESAMPLE}"

echo ">> Gerando datasets a partir de $CSV (resample='${RESAMPLE:-1min}')"
python scripts/cdd_to_finrl.py --csv "$CSV" --tic "${TIC:-BTCUSDT}" $RESAMPLE_ARG

echo ">> Treinando agentes"
python examples/FinRL_StockTrading_2026_2_train.py

echo ">> Backtest"
python examples/FinRL_StockTrading_2026_3_Backtest.py

echo ">> Concluído. Artefatos em trained_models/ e results/"
```

### 4.3 docker-compose.yml (raiz)

```yaml
services:
  finrl:
    image: ghcr.io/SEU_USUARIO/finrl:latest   # trocar SEU_USUARIO
    build: .
    command: ["bash", "scripts/run_experiment.sh"]   # orquestrador da 8.6/8.7
    environment:
      RESAMPLE: 5min        # granularidade
      N_ENVS: 6             # Opção A: envs paralelos (KVM 8 = 6; KVM 4 = 3)
      ORCAMENTO_SEG: 18000  # teto de tempo do treino (5 h) — maior que no Codespace
      TIC: BTCUSDT
    volumes:
      - ./data:/app/data                       # os 7 CSVs ficam AQUI (montados, não na imagem)
      - ./trained_models:/app/trained_models   # modelos/checkpoints persistidos
      - ./results:/app/results                 # tearsheet/gráficos/log
    restart: "no"           # tarefa "rodar e sair" (batch de treino)
```

> **Importante:** os CSVs **nunca entram na imagem Docker** (senão o build fica gigante). Ficam no `data/` do host, montados via volume. O `.dockerignore` (raiz) deve conter `data/` para o `COPY . .` não arrastá-los:
> ```
> data/
> trained_models/
> results/
> .git/
> ```

### 4.4 Testar local (no Codespace)

```bash
docker compose build
docker compose up          # roda o pipeline e sai
# subir/derrubar como serviço (se virar um Jupyter/API): up -d / down
```

**Subir e derrubar** (seu requisito 4):
```bash
docker compose up -d     # sobe em background
docker compose logs -f   # acompanha
docker compose down      # derruba tudo
```

---

## FASE 5 — CI/CD com GitHub Actions (1–2 h)

Dois workflows: **CI** (valida a cada push/PR) e **CD** (publica imagem e faz deploy na VPS).

### 5.1 CI — `.github/workflows/ci.yml`

```yaml
name: CI
on:
  push:
    branches: [ master, main ]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Deps de sistema
        run: sudo apt-get update && sudo apt-get install -y build-essential swig
      - name: Instalar
        run: |
          python -m pip install --upgrade pip
          pip install -e .
          pip install pytest
      - name: Lint (opcional)
        run: pip install black && black --check finrl scripts || true
      - name: Testes unitários do FinRL
        run: pytest unit_tests -q || true   # tire o "|| true" quando estabilizar
      - name: Smoke test do adaptador (sample pequeno + reamostragem)
        run: |
          python scripts/cdd_to_finrl.py \
            --csv data/raw/sample_btcusdt_1min.csv --tic BTCUSDT --resample 1h
          test -f train_data.csv && echo "adaptador OK"

  docker-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build imagem (sem publicar)
        run: docker build -t finrl:ci .
```

### 5.2 CD — `.github/workflows/deploy.yml`

Publica a imagem no **GHCR** e faz deploy na VPS via SSH. Dispara em push na `master` ou manualmente.

```yaml
name: Deploy
on:
  push:
    branches: [ master, main ]
  workflow_dispatch:

env:
  IMAGE: ghcr.io/${{ github.repository_owner }}/finrl

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - name: Login no GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build & push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ${{ env.IMAGE }}:latest,${{ env.IMAGE }}:${{ github.sha }}

  deploy-vps:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - name: Enviar imagem para a VPS (SEM iniciar o treino)
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.VPS_HOST }}
          username: ${{ secrets.VPS_USER }}
          key: ${{ secrets.VPS_SSH_KEY }}
          script: |
            cd ~/finrl
            echo ${{ secrets.GITHUB_TOKEN }} | docker login ghcr.io -u ${{ github.actor }} --password-stdin
            docker compose pull        # só baixa a última imagem; NÃO roda o treino
            docker image prune -f
```

> **Por que NÃO `docker compose up` aqui:** o container é um **batch de treino de horas**, não um serviço web. Se o deploy o iniciasse, todo merge na `master` dispararia um treino novo — e o job do Actions tem limite de ~6 h, então nem caberia. A regra é: **CI valida, CD apenas publica e entrega a imagem; o treino é disparado deliberadamente na VPS.**

### 5.3 Disparar o treino (deliberado, na VPS)

Depois que o CD entregou a imagem, você inicia o treino quando quiser, direto na VPS, em segundo plano:

```bash
ssh deploy@IP_DA_VPS
cd ~/finrl
# roda e sai; logs no arquivo; sobrevive ao fechar o SSH:
nohup docker compose run --rm finrl > results/run.log 2>&1 &
tail -f results/run.log        # acompanhar (Ctrl+C só sai do tail)
```

Vantagem da VPS sobre o Codespace: **não há idle timeout**. O treino roda até acabar, e os artefatos (modelos, tearsheet) ficam nos volumes montados. Se quiser automatizar treinos periódicos, dá para criar um workflow separado com `workflow_dispatch` (botão "Run") ou `schedule:` (cron) que faz esse mesmo `docker compose run` via SSH — mantendo-o **fora** do fluxo automático de todo push.

**Secrets a cadastrar** (repo → Settings → Secrets and variables → Actions):
- `VPS_HOST` — IP da VPS
- `VPS_USER` — usuário (ex.: `deploy`)
- `VPS_SSH_KEY` — chave **privada** SSH (a pública vai na VPS, ver Fase 6)

> `GITHUB_TOKEN` é automático. Para o `docker compose pull` funcionar na VPS, ou o pacote GHCR é público, ou você faz `docker login` na VPS (o script acima já loga).

---

## FASE 6 — Provisionar a VPS Hostinger e conectar o deploy (1 h)

### 6.1 Criar a VPS

1. Em `https://www.hostinger.com/br/servidor-vps`, contrate um plano **KVM**. Para a **Opção A (treino paralelo em CPU — Fase 8.7)**, o que manda é **número de vCPUs**; RAM tem que acompanhar para as cópias de env. Specs atuais (verificados em 2026, promoções mudam):
   - **KVM 8 — 8 vCPU / 32 GB / 400 GB NVMe (recomendado):** roda `N_ENVS=6` com folga de RAM. É o ponto ótimo para a Opção A.
   - **KVM 4 — 4 vCPU / 16 GB / 200 GB NVMe (econômico):** roda `N_ENVS=3` — já um salto enorme sobre o 1 env do Codespace.
   - KVM 2 (2 vCPU / 8 GB) só para testes leves; não vale para o treino sério.
   - São KVM reais (AMD EPYC, NVMe, recursos **dedicados**), então os vCPUs são de fato seus — bom para paralelismo.
2. Sistema operacional: **Ubuntu 24.04**. Datacenter **no Brasil** (menor latência). Se houver template com **Docker** pré-instalado, use-o (pula a 6.2).
3. Disco: os 7 CSVs (~centenas de MB) + datasets processados + imagem Docker (>2 GB) + modelos/checkpoints. Os 200–400 GB dos planos KVM 4/8 sobram.
4. Anote o **IP** e a senha root inicial.

> Nada de GPU aqui: a Opção A é CPU-pura, então o Dockerfile CPU da Fase 4 serve como está (sem CUDA).

### 6.2 Preparar a VPS (via terminal SSH)

Do Codespace ou de qualquer terminal:

```bash
ssh root@IP_DA_VPS

# usuário de deploy sem root
adduser deploy && usermod -aG sudo deploy

# Docker + Compose plugin
curl -fsSL https://get.docker.com | sh
usermod -aG docker deploy

# projeto na VPS
su - deploy
mkdir -p ~/finrl && cd ~/finrl
```

Copie para `~/finrl` na VPS o `docker-compose.yml` e a pasta `data/raw/` com os **7 CSVs anuais** (a imagem vem do GHCR, então o código-fonte não precisa estar lá). O jeito mais fácil é enviar **direto do Codespace**, onde os arquivos já estão:

```bash
# NO TERMINAL DO CODESPACE (os 7 CSVs já estão em data/raw/):
scp docker-compose.yml deploy@IP_DA_VPS:~/finrl/

# comprima os 7 de uma vez e envie num arquivo só (upload mais confiável):
tar czf btc_csvs.tgz -C data/raw .
scp btc_csvs.tgz deploy@IP_DA_VPS:~/finrl/
ssh deploy@IP_DA_VPS "mkdir -p ~/finrl/data/raw && tar xzf ~/finrl/btc_csvs.tgz -C ~/finrl/data/raw && rm ~/finrl/btc_csvs.tgz && ls -la ~/finrl/data/raw"
```

> Os CSVs sobem **uma vez**. Como ficam num volume montado, sobrevivem a `docker compose down`/`up` e a novos deploys da imagem. (Da sua máquina Windows também dá: `scp` no PowerShell ou WinSCP — mas do Codespace é menos passo.)

### 6.3 Chave SSH para o GitHub Actions

```bash
# no seu Codespace, gere um par dedicado ao deploy:
ssh-keygen -t ed25519 -f deploy_key -N ""

# pública -> autoriza na VPS:
ssh-copy-id -i deploy_key.pub deploy@IP_DA_VPS
#   (ou cole o conteúdo de deploy_key.pub em ~/.ssh/authorized_keys do deploy)

# privada -> cadastre como secret VPS_SSH_KEY no GitHub (conteúdo de 'deploy_key')
cat deploy_key
```

Nunca comite a chave privada. Apague `deploy_key*` do Codespace depois de cadastrar o secret.

### 6.4 Primeiro deploy

Faça um push na `master` (ou rode o workflow **Deploy** manualmente em Actions → Deploy → Run workflow). O Actions vai: buildar a imagem, publicar no GHCR e rodar `docker compose up -d` na VPS.

Verifique na VPS:
```bash
ssh deploy@IP_DA_VPS
cd ~/finrl && docker compose ps && docker compose logs -f
```

---

## FASE 7 — Operação do dia a dia (receita prática)

Como rodar treinos na VPS sem depender da memória da conversa. Assume o setup pronto (VPS provisionada, Docker, ponte SSH, imagem entregue pelo CD, 7 CSVs em `~/finrl/data/raw`).

### 7.1 Conectar na VPS (do Codespace)

```bash
ssh -i deploy_key deploy@187.127.4.38
cd ~/finrl
```

A chave `deploy_key` fica no Codespace (está no `.gitignore`, nunca vai para o Git). Se um dia recriar o Codespace, é preciso regenerar a chave e reautorizar (Fase 6.3).

### 7.2 Rodar um treino sério (robusto, sobrevive a fechar o SSH)

```bash
nohup docker compose run --rm finrl > results/run.log 2>&1 &
echo "PID: $!"
tail -f results/run.log
```

- `nohup ... &` desacopla o treino do terminal — pode fechar o SSH que ele continua rodando na VPS (diferente do Codespace, a VPS **não tem idle timeout**).
- `tail -f` acompanha o log ao vivo; **Ctrl+C sai só do tail**, não mata o treino.
- Ao terminar, o log fecha com um RESUMO FINAL e gera `results/tearsheet.html`.

> A 103 passos/s (N_ENVS=3), as 3 passadas completas (~1,4M passos) levam ~3h45. Ajuste o orçamento se quiser mais/menos (7.4).

### 7.3 Teste de fumaça antes de um run longo (2 min)

Sempre que mudar algo, valide o pipeline inteiro rapidinho antes de comprometer horas:

```bash
ORCAMENTO_SEG=120 docker compose run --rm finrl
```

O prefixo `ORCAMENTO_SEG=120` sobrepõe o teto só nesta execução (120 s). Se rodar do "Começo limpo" até o "RESUMO FINAL" sem erro, o caminho está saudável.

### 7.4 Ajustar os parâmetros (os "botões")

Três botões: `N_ENVS`, `ORCAMENTO_SEG`, `RESAMPLE`. Dois jeitos de girar:

- **Temporário (um run só):** prefixo na linha de comando, ex.:
  `RESAMPLE=15min ORCAMENTO_SEG=7200 docker compose run --rm finrl`
- **Permanente:** edite o `docker-compose.yml` na VPS (bloco `environment:`) — vale para todos os runs seguintes.

### 7.5 Começo limpo vs. retomar de checkpoint (atenção!)

Comportamento atual do `run_experiment.sh`: a etapa "(a) Começo limpo" **esvazia `trained_models/` a cada run**, então cada execução **começa do zero** — e o "resume automático" nunca acha checkpoint (é o que o log mostra: *"Nenhum checkpoint encontrado — treinando do zero"*). Ou seja, hoje **resume e começo-limpo se anulam**.

Para ter retomada real após uma queda (aproveitando os checkpoints salvos a cada 25k passos), o começo limpo precisa ser **opcional**. Correção recomendada (prompt para o Claude Code):

> No scripts/run_experiment.sh, torne a etapa "(a) Começo limpo" condicional a uma variável `FRESH` (`${FRESH:-0}`): só esvaziar trained_models/ e results/ quando `FRESH=1`. Com `FRESH=0` (default), preservar os checkpoints para o resume funcionar. Documente no topo do script.

Depois disso: `FRESH=1 docker compose run --rm finrl` para experimento novo; `docker compose run --rm finrl` (sem FRESH) para retomar de onde parou.

### 7.6 Pegar os resultados para olhar

Os artefatos ficam nos volumes montados da VPS (`~/finrl/results/` e `~/finrl/trained_models/`). Para ver o tear sheet, traga-o para o Codespace:

```bash
# NO CODESPACE:
scp -i deploy_key deploy@187.127.4.38:~/finrl/results/tearsheet.html .
```

Depois clique no `tearsheet.html` no explorador do VS Code (ou use "Open with Live Server"). Os CSVs de resultado (`equity_comparison.csv`, `weights_ppo.csv`) vêm do mesmo jeito.

### 7.7 Checar/parar um treino em andamento

```bash
docker compose ps          # o container 'finrl' está rodando?
docker compose logs -f     # logs (se não estiver usando o run.log)
docker compose down        # para e remove o container (interrompe o treino)
```

### 7.8 Ciclo para mudar código (o CI/CD na prática)

Toda mudança de **código** segue o fluxo que já exercitamos:

1. Ajuste no Codespace (você ou o Claude Code), num branch (`feat/...`).
2. `git add ... && git commit -m "..." && git push` → o **CI** valida.
3. Abra o Pull Request **apontando para `dangerousavoid/master`** (nunca `AI4Finance-Foundation`) e faça o **merge**.
4. O merge dispara o **Deploy**, que reconstrói a imagem e a entrega na VPS.
5. Na VPS: `docker compose pull` para pegar a imagem nova, e rode (7.2/7.3).

Mudança só de **parâmetro** (N_ENVS, orçamento, resample) não precisa de deploy — é o 7.4.

### 7.9 Novos dados

Baixe CSVs atualizados da CDD, envie para `~/finrl/data/raw/` na VPS (o `tar`+`scp` da Fase 6.2) e rode de novo. Para treinos periódicos automáticos, dá para um workflow com `schedule:` (cron) que faz `docker compose run` via SSH — mantendo-o **fora** do fluxo de todo push.

---


## FASE 8 — Consumir o resultado: baseline honesto, métricas e inferência (2–3 h)

Esta fase transforma "modelos treinados" em respostas úteis. Três entregas, todas em **arquivos novos** (não refatoram o FinRL):

1. `scripts/evaluate.py` — compara os agentes contra **buy-and-hold do BTC** e imprime métricas (Sharpe, drawdown, retorno).
2. `scripts/infer.py` — carrega um modelo treinado e devolve a **ação para a barra mais recente**.
3. Nota sobre anualização de métricas com dados intradiários.

> **Por que não usar o Part 4–7 do exemplo?** O script de backtest de exemplo compara com **Mean-Variance Optimization** e o índice **Dow Jones** (`^DJI` via yfinance) — ambos são baselines de mercado de **ações** e não fazem sentido para BTC (MVO num único ativo é degenerado; o Dow é irrelevante). Além disso, o exemplo cria o env com `turbulence_threshold=70, risk_indicator_col="vix"` — e como nossos dados de cripto **não têm** coluna `vix`, isso quebraria. Por isso substituímos por um script próprio.

### 8.1 `scripts/evaluate.py` — agentes vs. buy-and-hold + métricas

```python
from __future__ import annotations
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.plot import backtest_stats

MODEL_CLASSES = {"a2c": A2C, "ddpg": DDPG, "ppo": PPO, "td3": TD3, "sac": SAC}
INITIAL_AMOUNT = 1_000_000


def build_env(trade: pd.DataFrame) -> StockTradingEnv:
    stock_dim = len(trade.tic.unique())
    state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
    # IMPORTANTE p/ cripto: turbulence_threshold=None -> não exige coluna vix/turbulence
    return StockTradingEnv(
        df=trade,
        turbulence_threshold=None,
        hmax=100,
        initial_amount=INITIAL_AMOUNT,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[0.001] * stock_dim,
        sell_cost_pct=[0.001] * stock_dim,
        state_space=state_space,
        stock_dim=stock_dim,
        tech_indicator_list=INDICATORS,
        action_space=stock_dim,
        reward_scaling=1e-4,
    )


def buy_and_hold(trade: pd.DataFrame) -> pd.DataFrame:
    """Equity de comprar BTC na 1ª barra de trade e segurar."""
    prices = trade.groupby("date")["close"].first().sort_index()
    equity = INITIAL_AMOUNT * (prices / prices.iloc[0])
    return pd.DataFrame({"date": equity.index, "account_value": equity.values})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trade", default="trade_data.csv")
    p.add_argument("--agents", nargs="+", default=["ppo", "a2c", "sac"])
    args = p.parse_args()

    trade = pd.read_csv(args.trade)

    curves = {}
    for name in args.agents:
        model = MODEL_CLASSES[name].load(f"{TRAINED_MODEL_DIR}/agent_{name}")
        env = build_env(trade)
        df_acc, df_act = DRLAgent.DRL_prediction(model=model, environment=env)
        df_acc = df_acc.set_index("date")["account_value"]
        curves[name] = df_acc

        print(f"\n===== Métricas: {name.upper()} =====")
        print(backtest_stats(account_value=df_acc.reset_index()))
        df_act.to_csv(f"results/actions_{name}.csv")

    bh = buy_and_hold(trade).set_index("date")["account_value"]
    curves["buy_and_hold"] = bh
    print("\n===== Métricas: BUY & HOLD BTC =====")
    print(backtest_stats(account_value=bh.reset_index()))

    result = pd.DataFrame(curves)
    result.to_csv("results/equity_comparison.csv")

    plt.figure(figsize=(15, 6))
    result.plot(ax=plt.gca())
    plt.title("Agentes DRL vs. Buy & Hold (BTC)")
    plt.xlabel("Tempo"); plt.ylabel("Valor da carteira ($)")
    plt.savefig("results/evaluate_result.png", dpi=150, bbox_inches="tight")
    print("\nGráfico salvo em results/evaluate_result.png")


if __name__ == "__main__":
    main()
```

Rodar:

```bash
mkdir -p results
python scripts/evaluate.py --trade trade_data.csv --agents ppo a2c sac
```

Saídas: `results/evaluate_result.png` (curvas comparadas), `results/equity_comparison.csv` (equity de todos), `results/actions_<agente>.csv` (série de decisões de cada agente) e, no console, a tabela de métricas da pyfolio (retorno anual, Sharpe, Sortino, max drawdown, volatilidade, etc.).

**A leitura que importa:** se a curva de um agente não fica **acima** da curva de buy-and-hold do BTC, a política de RL não agregou valor sobre "simplesmente comprar e segurar". Esse é o teste de mérito do projeto.

### 8.2 `scripts/infer.py` — ação para a barra mais recente

Depois de validado, o valor prático é: dado o BTC **agora**, o que o modelo faria? O script processa as barras recentes com o mesmo adaptador, roda a política e devolve a última ação.

```python
from __future__ import annotations
import argparse
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from scripts.cdd_to_finrl import load_cdd

MODEL_CLASSES = {"a2c": A2C, "ddpg": DDPG, "ppo": PPO, "td3": TD3, "sac": SAC}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV recente da CDD (pode ser um recorte)")
    p.add_argument("--agent", default="ppo")
    p.add_argument("--resample", default="1h")
    p.add_argument("--tic", default="BTCUSDT")
    # estado atual da carteira (o env precisa saber onde você está HOJE):
    p.add_argument("--cash", type=float, default=1_000_000)
    p.add_argument("--held", type=float, default=0.0, help="qtde de BTC atualmente em carteira")
    args = p.parse_args()

    df = load_cdd(args.csv, tic=args.tic, resample=args.resample)
    fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS,
                         use_vix=False, use_turbulence=False, user_defined_feature=False)
    processed = fe.preprocess_data(df)

    stock_dim = 1
    state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
    env = StockTradingEnv(
        df=processed, turbulence_threshold=None,
        hmax=100, initial_amount=args.cash, num_stock_shares=[args.held],
        buy_cost_pct=[0.001], sell_cost_pct=[0.001],
        state_space=state_space, stock_dim=stock_dim,
        tech_indicator_list=INDICATORS, action_space=stock_dim, reward_scaling=1e-4,
    )

    model = MODEL_CLASSES[args.agent].load(f"{TRAINED_MODEL_DIR}/agent_{args.agent}")
    _, df_actions = DRLAgent.DRL_prediction(model=model, environment=env)

    last = df_actions.tail(1)
    qty = float(last["actions"].iloc[0]) if "actions" in last else float(last.iloc[0, -1])
    verbo = "COMPRAR" if qty > 0 else ("VENDER" if qty < 0 else "MANTER")
    print(f"\nBarra mais recente: {last.index[-1] if last.index.name else last['date'].iloc[0]}")
    print(f"Agente {args.agent.upper()} sugere: {verbo}  (ação bruta = {qty:.4f} unidades)")


if __name__ == "__main__":
    main()
```

Rodar:

```bash
python scripts/infer.py --csv data/raw/btcusdt_1min.csv --agent ppo --resample 1h \
  --cash 1000000 --held 0
```

> **Por que passar `--cash` e `--held`?** A decisão do agente depende do **estado da carteira** (quanto de caixa e de BTC você já tem), não só do preço. Para uma decisão realista "para agora", informe seu caixa e sua posição atuais. Se deixar os defaults, você obtém a decisão como se estivesse começando do zero — útil para inspeção, não para operar.

> **Isto NÃO envia ordens.** É só a decisão. Ligar isso a uma corretora/exchange (via Alpaca no FinRL clássico, ou `ccxt`/Binance para cripto) é o passo de "colocar capital", que exige **paper trading primeiro** e está fora deste plano. Nada aqui é recomendação de investimento.

### 8.3 Cuidado com métricas em dados intradiários

A `backtest_stats` usa a **pyfolio**, que assume que a série de retornos é **diária** e anualiza com fator de ~252. Com barras de minuto/hora, o retorno "por barra" não é diário, então **Sharpe e retorno anual sairão distorcidos**. Duas saídas:

- **Simples e correta:** reamostre a curva de equity para diária antes de medir. Ex.: converta `account_value` para série temporal indexada por data e faça `.resample("1D").last()`; passe isso à `backtest_stats`. Aí a anualização faz sentido.
- **Para comparar agentes entre si**, a distorção afeta todos igualmente, então o **ranking** relativo continua válido mesmo sem reamostrar — só não leia os números absolutos como "retorno anual real".

### 8.4 Encaixe no pipeline e no CI

- Adicione ao `scripts/run_pipeline.sh`, após o treino, uma chamada a `python scripts/evaluate.py` para que cada execução na VPS já produza `results/evaluate_result.png` e as métricas.
- No CI, um smoke test rápido: rodar `evaluate.py` sobre o `sample_` pequeno com pouquíssimos timesteps, só para garantir que a avaliação não quebrou.

---

## FASE 8.5 (OPCIONAL) — Backtester "sério" desacoplado (2–3 h)

Objetivo: ter métricas de nível profissional (tear sheet, custos, turnover, comparação com benchmark) **sem** enxertar código do FinRL-X e sem virar Frankenstein. A chave é uma regra de arquitetura:

> **Dependência de mão única, através de um contrato de dados.** Nada aqui importa o FinRL-X. Roubamos dele apenas a *ideia* "weight-centric": o único contrato entre "quem decide" e "quem avalia/executa" é uma série temporal `date → weight` (a fração do capital em BTC, em `[0, 1]`; use `[-1, 1]` se admitir posição vendida).

Assim, quem produz o `weight` hoje é o FinRL do Plano A; amanhã pode ser uma regra manual, outro modelo, o que for — o backtester **não muda**, porque ele só conhece `{preços, pesos}`.

Dois arquivos novos, sem tocar em nada existente:

```
scripts/
  signals.py        # NOVO: FinRL df_actions -> contrato date->weight
  backtest_pro.py   # NOVO: recebe {preços, pesos}; ZERO import de FinRL/FinRL-X
```

### 8.5.1 `scripts/signals.py` — extrai o contrato (weight) a partir do FinRL

Esta é a **única** parte que conhece o formato do FinRL. Ela reconstrói a posição (soma acumulada das ações) e a divide pelo patrimônio para obter o peso limpo.

```python
from __future__ import annotations
import argparse
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import TRAINED_MODEL_DIR
from scripts.evaluate import build_env, INITIAL_AMOUNT  # reuso do seu próprio código (Fase 8)

MODEL_CLASSES = {"a2c": A2C, "ddpg": DDPG, "ppo": PPO, "td3": TD3, "sac": SAC}


def extract_weights(trade: pd.DataFrame, agent: str, held0: float = 0.0) -> pd.DataFrame:
    model = MODEL_CLASSES[agent].load(f"{TRAINED_MODEL_DIR}/agent_{agent}")
    env = build_env(trade)
    df_acc, df_act = DRLAgent.DRL_prediction(model=model, environment=env)

    # preços (close) por barra
    prices = trade.groupby("date")["close"].first().sort_index()

    # ações -> posição acumulada (cotas de BTC mantidas)
    acts = df_act.copy()
    if "date" in acts.columns:
        acts = acts.set_index("date")
    act_col = "actions" if "actions" in acts.columns else acts.columns[0]
    holdings = held0 + acts[act_col].cumsum()

    # patrimônio vindo do próprio env
    equity = df_acc.set_index("date")["account_value"]

    df = pd.DataFrame({"close": prices, "holdings": holdings, "equity": equity}).dropna()
    df["weight"] = (df["holdings"] * df["close"] / df["equity"]).clip(-1, 1)
    return df.reset_index().rename(columns={"index": "date"})[["date", "weight"]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trade", default="trade_data.csv")
    p.add_argument("--agent", default="ppo")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    trade = pd.read_csv(args.trade)
    w = extract_weights(trade, args.agent)
    out = args.out or f"results/weights_{args.agent}.csv"
    w.to_csv(out, index=False)
    print(f"contrato salvo em {out}  ({len(w)} barras, peso médio={w.weight.mean():.3f})")


if __name__ == "__main__":
    main()
```

> **Por que este é o único ponto "sujo":** ações do FinRL vêm em cotas escaladas por `hmax` e normalizadas. Aqui elas são traduzidas para um peso agnóstico (fração do patrimônio). Todo o resto do fluxo nunca mais precisa saber que o FinRL existiu.

### 8.5.2 `scripts/backtest_pro.py` — recebe {preços, pesos}, mede tudo

Zero acoplamento: não importa FinRL nem FinRL-X. Aplica o peso de forma **causal** (sem look-ahead), desconta **custos por turnover** e gera um tear sheet com a `quantstats`.

```python
from __future__ import annotations
import argparse
import pandas as pd
import quantstats as qs


def run_backtest(prices: pd.Series, weights: pd.Series, fee: float = 0.001,
                 to_daily: bool = False):
    idx = prices.index.intersection(weights.index)
    prices, weights = prices.loc[idx].sort_index(), weights.loc[idx].sort_index()

    ret = prices.pct_change().fillna(0.0)
    pos = weights.shift(1).fillna(0.0)          # SEM look-ahead: peso de t-1 no retorno de t
    turnover = pos.diff().abs().fillna(0.0)
    strat_ret = pos * ret - turnover * fee      # custo proporcional ao giro
    bh_ret = ret                                # benchmark = buy & hold (peso 1)

    strat_ret.index = pd.to_datetime(strat_ret.index)
    bh_ret.index = pd.to_datetime(bh_ret.index)

    if to_daily:  # corrige anualização em dados intradiários (ver nota 8.5.4)
        strat_ret = (1 + strat_ret).resample("1D").prod() - 1
        bh_ret = (1 + bh_ret).resample("1D").prod() - 1

    return strat_ret, bh_ret


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trade", default="trade_data.csv")
    p.add_argument("--weights", required=True, help="CSV date,weight (saída do signals.py)")
    p.add_argument("--fee", type=float, default=0.001, help="custo por unidade de turnover")
    p.add_argument("--to-daily", action="store_true", help="reamostra p/ diária antes das métricas")
    p.add_argument("--out", default="results/tearsheet.html")
    args = p.parse_args()

    trade = pd.read_csv(args.trade)
    prices = trade.groupby("date")["close"].first().sort_index()
    weights = pd.read_csv(args.weights).set_index("date")["weight"].sort_index()

    strat, bh = run_backtest(prices, weights, fee=args.fee, to_daily=args.to_daily)

    print(qs.reports.metrics(strat, benchmark=bh, mode="full", display=False))
    qs.reports.html(strat, benchmark=bh, output=args.out,
                    title="Estratégia DRL vs Buy & Hold (BTC)")
    print(f"\nTear sheet salvo em {args.out}")


if __name__ == "__main__":
    main()
```

Rodar (dois passos, contrato no meio):

```bash
pip install quantstats
python scripts/signals.py --trade trade_data.csv --agent ppo         # gera results/weights_ppo.csv
python scripts/backtest_pro.py --weights results/weights_ppo.csv --to-daily
```

Saídas: `results/weights_<agente>.csv` (o contrato) e `results/tearsheet.html` (Sharpe, Sortino, max drawdown, CAGR, volatilidade, comparação com o buy-and-hold, gráficos). No console, a tabela completa de métricas.

### 8.5.3 O que torna o backtest "sério" (e por que o contrato limpo obriga a encará-lo)

Não é a biblioteca — são três cuidados que o desacoplamento força você a tratar de frente:

- **Look-ahead:** sempre `weights.shift(1)`. O env do FinRL já é causal, mas ao extrair o sinal e re-simular por fora, é você quem garante que a decisão de `t` só afeta o retorno de `t → t+1`.
- **Custos e slippage:** em dados de minuto, taxas de taker (~0,04–0,1%) e derrapagem definem se a estratégia sobrevive. Ajuste `--fee`.
- **Turnover:** um sinal de minuto que troca de posição toda hora pode brilhar no bruto e perder no líquido. O termo `turnover * fee` expõe exatamente isso.

### 8.5.4 Cuidado com anualização intradiária (mesmo alerta da 8.3)

A `quantstats`, como a pyfolio, assume retornos **diários** para anualizar. Com barras de minuto/hora, use `--to-daily` (o script reamostra os retornos para diária compondo-os antes de medir) para que Sharpe e CAGR façam sentido. Sem isso, o *ranking* entre estratégias ainda é válido, mas os números absolutos não.

### 8.5.5 O ganho de arquitetura (o "anti-Frankenstein")

Como `backtest_pro.py` só conhece `{preços, pesos}`:

- Você pode alimentá-lo com pesos de **qualquer** origem — uma regra manual, um segundo modelo, um ensemble — trocando só o produtor do contrato, nunca o backtester.
- Este módulo é uma "fatia greenfield" isolada: é exatamente o tipo de componente que o Spec Kit da Fase 9 trataria bem (spec curta, escopo fechado).
- Alternativas à `quantstats`, se quiser: **`backtesting.py`** (feito para um único ativo com OHLC e gráfico interativo — encaixe natural para "só BTC") ou **`bt`** (o que o FinRL-X usa; mais voltado a portfólio). O contrato `date→weight` serve para os três; só muda o adaptador de saída.

> Nada aqui envia ordens nem é recomendação de investimento — é avaliação offline.

---

## FASE 8.6 — Experimento sério em 5 minutos (calibrado)

Configuração do primeiro treino "de verdade" (não o smoke test). Granularidade de **5 min**, indicadores multi-horizonte por janela deslizante (diário + semanal), split honesto com validação, e o backtest robusto da 8.5.

### Parâmetros

| Parâmetro | Valor | Porquê |
|---|---|---|
| Granularidade | `--resample 5min` | escolha do projeto |
| Split treino | 2020-01-01 00:00:00 → 2024-06-30 23:59:00 | ~4,5 anos, vários regimes (bull 20-21, bear 22, recuperação 23-24) |
| Split validação | 2024-07-01 00:00:00 → 2025-06-30 23:59:00 | usado só para **escolher** o modelo |
| Split trade (teste) | 2025-07-01 00:00:00 → última data dos dados | held-out puro; detectar o máximo automaticamente |
| `hmax` | 15 | com $1M e BTC ~$60k dá ~16 BTC; o default 100 é absurdo p/ cripto |
| `total_timesteps` | 3 × nº de linhas do train (~1,4M) | ≈ 3 passadas; subir depois se a validação pedir |
| Algoritmo | só **PPO** por enquanto | on-policy, estável, mais rápido em CPU |
| Custo backtest | `--fee 0.001` (placeholder) | confirmar a taxa taker da sua exchange |

### Indicadores (lista congelada — `INDICATORS` no `config.py`)

Janelas contadas em **barras de 5 min**: 1 dia ≈ 288, 1 semana ≈ 2016.

```
["macd","rsi_30","boll_ub","boll_lb","cci_30","dx_30",   # intradiário (timing fino)
 "close_288_sma","rsi_288",                              # ~diário
 "close_2016_sma","rsi_2016"]                            # ~semanal
```

Mensal (~8640 barras) ficou **de fora**: exigiria ~30 dias de warmup, ficaria quase constante e o ganho é duvidoso. Optamos por janela deslizante simples (nativa do FinRL), **não** por empilhamento real de timeframes (que exigiria reamostragem + `shift(1)` + merge as-of para evitar look-ahead) — fica registrado como caminho possível, porém descartado por ora.

> **Trave a lista antes de treinar.** A dimensão de entrada do modelo depende de `len(INDICATORS)`; mudar a lista (ou a granularidade) invalida o `.zip` salvo e obriga re-treino do zero. Use a mesma lista em treino, avaliação e inferência.

### Metodologia (o que separa "teste sério" de auto-engano)

Treinar no **train**; escolher o modelo/checkpoint pelo desempenho no **val**; só então reportar o número final no **trade**. O conjunto de trade nunca entra em nenhuma decisão de ajuste — senão você otimiza para o próprio teste.

### Calibrar o tempo (como teto automático, não como aprovação)

Um script orquestrador (`scripts/run_experiment.sh`) faz a calibração e dimensiona o treino sozinho: treina 20k passos, mede passos/segundo, e define `total_timesteps = min(3 × len(train), passos_por_segundo × ORÇAMENTO)`, com `ORÇAMENTO` = 3 h por padrão (variável editável no topo do script). Assim o run nunca estoura o tempo previsto e não exige aprovação manual. Se nem uma passada couber no orçamento, o script emite um WARNING recomendando `--resample 15min` ou a VPS da Fase 6, mas prossegue.

### Cuidados operacionais (porque não há VPS — Fases 6/7 puladas)

- **Idle timeout do Codespace → 240 min** (Settings → Codespaces).
- **`CheckpointCallback` a cada 100k passos** + **RESUME** (retomar do último checkpoint): se o Codespace dormir, o run continua de onde parou.
- Rodar em segundo plano: `nohup bash scripts/run_experiment.sh > results/run.log 2>&1 &` e acompanhar com `tail -f results/run.log`. Runs de muitas horas seriam, a rigor, trabalho para a VPS da Fase 6.

### Report multi-período

"Métricas diárias/semanais/mensais" saem do `backtest_pro.py --to-daily` (reamostra os retornos para diária → anualização correta; o tear sheet da quantstats já traz o mapa mensal e o resumo anual). Isso é camada de **report**, independente dos indicadores (que são camada de **feature**).

### Execução autônoma (Claude Code)

O prompt completo e copiável está no chat. Diferente das fases anteriores, este roda **sem aprovações**: o agente implementa tudo e escreve um único `scripts/run_experiment.sh` que encadeia dataset → calibração → treino (com orçamento de tempo) → validação → backtest 8.5, lançado em segundo plano com `nohup` e checkpoints/resume. No Claude Code, usar "auto-accept edits" (Shift+Tab) ou `claude --dangerously-skip-permissions` (seguro no sandbox do Codespace). Sem tocar em `finrl/`.

---

## FASE 8.7 — Treino paralelo em CPU (Opção A: escalar throughput sem GPU)

Diagnóstico da 8.6: os **38 passos/s** no Codespace não vêm da rede neural — vêm do `step()` do ambiente (pandas), que roda em **CPU**. Logo, GPU ajudaria pouco; o ganho real vem de rodar **vários ambientes em paralelo** com `SubprocVecEnv` do stable-baselines3, um por núcleo. Numa VPS de 8 vCPUs isso pode dar ~6× o throughput — várias passadas completas dentro do orçamento de tempo, que é o que faltou na 8.6.

Toda a mudança é em `scripts/` (o treino); **não** toca em `finrl/`.

### 8.7.1 A mudança: fábrica de env + `SubprocVecEnv`

O SB3 sobe N processos, cada um com uma cópia independente do env. PPO (on-policy) se beneficia diretamente: coleta rollouts dos N envs a cada iteração.

```python
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

def make_env(train_df, rank: int):
    """Fábrica: cada processo recebe uma instância independente do StockTradingEnv."""
    def _init():
        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
        env = StockTradingEnv(df=train_df, **ENV_KWARGS)   # turbulence_threshold=None etc.
        env.reset(seed=rank)      # semente distinta por processo
        return env
    return _init

def build_vec_env(train_df, n_envs: int):
    if n_envs <= 1:
        # caminho single-env (Codespace / debug)
        return build_env(train_df)
    venv = SubprocVecEnv([make_env(train_df, i) for i in range(n_envs)])
    return VecMonitor(venv)       # VecMonitor preserva as métricas de episódio
```

No treino, o `total_timesteps` continua sendo o **total somado entre os envs** — então a mesma meta termina ~N× mais rápido (menos overhead). O `CheckpointCallback` e o resume da 8.6 seguem funcionando.

### 8.7.2 Parametrização (para o compose controlar)

`run_experiment.sh` e `train.py` devem ler variáveis de ambiente com defaults, para o `docker-compose.yml` ajustar sem rebuild:

- `N_ENVS` (default 1 no Codespace; **6** na VPS KVM 8) — nº de envs paralelos.
- `ORCAMENTO_SEG` (default 10800) — teto de tempo; na VPS dá para subir.
- `RESAMPLE` (ex.: `5min`).

Regra de dimensionamento: **N_ENVS = nº de vCPUs − 1 ou 2** (deixe núcleos para o processo principal e o SO). KVM 8 (8 vCPU) → `N_ENVS=6`. KVM 4 (4 vCPU) → `N_ENVS=3`.

### 8.7.3 Duas armadilhas do paralelismo

- **Memória.** Cada env carrega sua cópia do DataFrame de treino. Com 5 min (~473k linhas × 17 colunas) são ~centenas de MB por env; 6 envs cabem folgados nos 32 GB do KVM 8, mas **no Codespace (8 GB) não tente N_ENVS alto** — daria OOM. Por isso o default 1 no Codespace.
- **Recalibrar.** Os passos/s da 8.6 foram medidos com 1 env. Na VPS, com N envs, o `--calibrate` precisa medir o throughput **agregado** (passos totais ÷ tempo) para o orçamento fazer a conta certa. É só rodar a calibração de novo na VPS.

### 8.7.4 Prompt para o Claude Code

> Leia CLAUDE.md e as Fases 8.6/8.7 de docs/plano_finrl_btc.md. Adicione treino paralelo (Opção A) SEM tocar em finrl/:
> 1. Em train.py, crie `make_env(train_df, rank)` e `build_vec_env(train_df, n_envs)` usando SubprocVecEnv + VecMonitor; n_envs<=1 mantém o caminho single-env atual.
> 2. `N_ENVS`, `ORCAMENTO_SEG` e `RESAMPLE` passam a ser lidos de variáveis de ambiente (com defaults; N_ENVS default 1).
> 3. A calibração deve medir throughput AGREGADO dos N envs.
> 4. CheckpointCallback e resume continuam funcionando com o VecEnv.
> Valide localmente com N_ENVS=2 num recorte pequeno e me mostre passos/s agregado vs single-env.

---

## FASE 8.8 — Aprendizado real: recompensa com risco, ensemble e timeframe 1h

Diagnóstico da 8.6/8.7 + leitura do paper do FinRL: com um ativo só, recompensa de retorno puro e teste em bull market, "comprar e segurar" é a política ótima — o agente aprendeu exatamente o que pedimos. Para haver timing real, é preciso: (a) motivo matemático para sair (recompensa penalizando risco), (b) features de regime/volatilidade e volume no estado, (c) exploração, (d) teste contendo uma queda fora da amostra.

### Decisões (e porquês)

- **Timeframe 1h** (não 5min): sinal/ruído 12× melhor para fenômenos de dias/semanas; custos de giro deixam de dominar; ~57k linhas tornam ensemble+sementes+15 passadas viáveis na KVM 4. Diário (~2,4k barras) é pouco para DRL. 5min fica para refinar execução no futuro.
- **Splits:** treino 2020-01-01→2023-12-31 (bear de 2022 DENTRO do treino), val 2024, teste 2025-01-01→fim (contém o crash ~30% de abr/2025 — prova de fogo out-of-sample).
- **Recompensa (scripts/risk_env.py, subclasse do StockTradingEnv — aditivo):** `r = Δv − λ·ΔDD`, onde ΔDD = aumento do drawdown (pico − valor, só a parcela nova). λ via env var; λ=0 reproduz o env original (controle).
- **INDICATORS (1h):** `macd, rsi_14, wr_14, atr_14, close_24_mstd, adx, vr_26, mfi_14, close_24_sma, close_168_sma` — cobre tendência, momentum, **volatilidade/regime** e **volume** (os dois buracos do set anterior). No adaptador, converter para razões estacionárias: `close/sma−1`, `atr/close`, `mstd/close`.
- **Exploração:** `ent_coef=0.01` no PPO/A2C (default 0 = sem exploração → congela em buy-and-hold).
- **Passadas:** 15 (≈525k passos/agente em ~35k linhas de treino). 2 sementes nos braços principais.
- **Braços:** PPO-λ0 (controle/diagnóstico), PPO-λ, A2C-λ, SAC-λ (SAC no lugar do DDPG do paper: exploração por entropia, DDPG é frágil). SAC é off-policy: ~2–3× mais lento por passo em CPU.
- **Ensemble (paper [51], versão estática):** escolher o braço com melhor Sharpe na VALIDAÇÃO; adicionalmente, comitê por média do contrato date→weight (via 8.5). Ambos reportados no teste.
- **Critério de sucesso:** no crash de abr/2025 (teste), max drawdown do agente-λ visivelmente menor que buy-and-hold e que o controle-λ0, com a série de pesos mostrando redução de posição. Retorno total NÃO é o critério primário deste ciclo.

### Orçamento estimado (KVM 4, N_ENVS=3)

Em 1h de barra o env roda mais leve; PPO/A2C ~1–2h por braço, SAC ~3–5h. Total com sementes: uma noite (ORCAMENTO_SEG por braço: 14400, folga). Rodar via nohup + run.log (Fase 7.2), com FRESH=1 no primeiro braço.

### Saídas

`results/` com: métricas por braço (val e teste, reamostradas p/ diária), tearsheet do vencedor e do comitê, `weights_<braço>.csv`, e um `summary.csv` comparando retorno/Sharpe/maxDD de todos os braços vs buy-and-hold no teste.

### Resultado da 8.8 (executado)

- **UC01 (5min, 8.6/8.7):** agente colapsou em ~buy-and-hold (empate, +0,36% num bull). Validou pipeline + paralelismo (38→103 passos/s).
- **8.8 λ=0.5:** todos os braços com risco colapsaram para **caixa** (peso 0). Recompensa de risco forte demais.
- **8.8 varredura λ∈{0, 0.05, 0.1, 0.2} (PPO):** **penhasco tudo-ou-nada** — λ≤0.05 → segura tudo (peso 0.94); λ≥0.1 → foge (peso 0). Não há meio-termo estável.
- O teste (jan/2025→jul/2026) foi de **queda** (BTC −32%); o controle acompanhou (−31%, maxDD −51%).

---

## FASE 8.9 — Correção pela AÇÃO (contínua) + protocolo honesto (guiado por pesquisa)

Um levantamento cético do estado da arte (jul/2026) explicou nossos becos e reordenou a rota. Fontes-chave: Moody & Saffell 1998/2001; Zhang, Zohren & Roberts 2020 (arXiv 1911.10107); Hambly/Xu/Yang 2023; Sun/Wang/An 2023; Bailey & López de Prado 2014 / López de Prado 2018; Borrageiro et al. 2022; Bandarupalli 2025.

### Achados que mudam o plano (decisões)

- **D-A (a mais importante): a raiz do tudo-ou-nada é o ESPAÇO DE AÇÃO, não a recompensa.** O `StockTradingEnv` opera em **cotas** → decisões de canto. Correção: **ação contínua** de posição-alvo (`Box`). Só depois mexer na recompensa. Teste decisivo: se colapsar mesmo com ação contínua, aí sim a culpa é da recompensa.
- **D-B: o colapso para buy-and-hold é o ótimo conhecido** em tendência de alta (não é bug). Timing de ativo único que bate B&H **não tem evidência robusta**; o caso "vencedor" em cripto é 71% *funding rate* (não timing spot). **Resultado honesto é entregável válido.**
- **D-C: penalizar VARIÂNCIA, não drawdown absoluto.** Drawdown não é policy-invariant → penhasco (confirmado por nós). Usar **Sharpe diferencial** (Moody & Saffell) ou `μ−(λ/2)σ²`. Varredura de λ **logarítmica** (1e-5, 1e-4, 1e-3), não 0.05–0.2.
- **D-D: custos realistas DESDE o treino** (~5–10 bps/giro; testar 5/10/25) + regularização de turnover. Sem isso, giro é ilusório.
- **D-E: validação honesta obrigatória** — walk-forward + CPCV purgado/embargo; **5–10 sementes** (média±desvio); **Deflated Sharpe Ratio**; baseline sempre B&H líquido + momentum.
- **D-F: features externas (Fear & Greed, on-chain, funding) só depois**, via ablação, point-in-time + embargo (look-ahead).
- **D-G (alternativa se DRL não convergir):** RL imitativo (imitar oráculo ex-post) ou híbrido detecção-de-regime + volatility targeting.

### Sequência de experimentos (reordenada pela evidência)

1. **Ação contínua** de posição-alvo (`Box[0,1]`) — este passo. Isola o efeito do espaço de ação; recompensa segue sendo retorno (líquido de custos).
2. Recompensa que penaliza variância (Sharpe diferencial / média-variância) + varredura log de λ.
3. Custos desde o treino + regularização de turnover.
4. (transversal) Protocolo de validação honesta (múltiplas sementes, DSR, walk-forward).
5. Features externas por ablação. 6. Se nada convergir: alternativa D-G.

### Experimento 1 — `scripts/target_weight_env.py` (novo, aditivo; não toca em `finrl/`)

- Ação **contínua** `Box(0,1)` = fração-alvo do patrimônio em BTC (parametrizar p/ [−1,1] futuro).
- A cada passo: ajusta exposição até `w_t`, aplica custo proporcional ao turnover `|w_t−w_{t-1}|` (default 10 bps, `COST_BPS`), avança uma barra.
- Estado: mesmas INDICATORS + posição/caixa normalizados (compatível com `train.py`).
- Recompensa: **só** Δv líquido de custos (sem penalização de risco ainda — isso é o passo 2).
- `train.py`: flag `--env {stocktrading,target_weight}` (default sem regressão); mantém N_ENVS/checkpoints/resume/seeds.
- `signals.py`: com env contínuo, o peso É a própria ação `w_t` (não reconstruído de cotas).
- `run_ensemble.sh`: variável `ENV_KIND` (default stocktrading) repassada como `--env`.
- **Smoke test decisivo:** rodar `--env=target_weight` num recorte pequeno e medir **desvio-padrão dos pesos**. Espalhados (0.3, 0.6, 0.8…) = ação contínua destravou o meio-termo. Colados em 0/1 = a raiz é a recompensa → ir ao passo 2.

---

## Checklist rápido

- [ ] Fork do FinRL criado
- [ ] `.devcontainer/` commitado e Codespace subindo
- [ ] `pip install -e .` funcionando (import OK)
- [ ] Conta gratuita na CDD criada; CSV de **1 minuto** do BTC baixado em `data/raw/`
- [ ] CSV grande no `.gitignore`; só o `sample_*` versionado
- [ ] `scripts/cdd_to_finrl.py` gerando `train_data.csv` / `trade_data.csv` (testado com `--resample`)
- [ ] Datas (com timestamp) e `timesteps` de treino ajustados em `finrl/config.py` / scripts de exemplo
- [ ] Treino + backtest rodando com dados de BTC (primeiro num recorte/reamostrado)
- [ ] `.dockerignore` com `data/`; `Dockerfile` + `docker-compose.yml` + `run_pipeline.sh` OK (`docker compose up`)
- [ ] `ci.yml` verde
- [ ] VPS Hostinger criada com Docker (dimensionada p/ RAM se for 1min puro)
- [ ] CSV grande enviado à VPS via `scp` (comprimido) para o volume `data/`
- [ ] Secrets (`VPS_HOST`, `VPS_USER`, `VPS_SSH_KEY`) cadastrados
- [ ] `deploy.yml` publicando no GHCR e subindo na VPS
- [ ] `scripts/evaluate.py` comparando agentes vs. buy-and-hold + métricas (pyfolio)
- [ ] `scripts/infer.py` devolvendo a ação da barra mais recente
- [ ] Métricas reamostradas para diária (se usar dados intradiários)
- [ ] `evaluate.py` encaixado no `run_pipeline.sh` e no CI (smoke test)
- [ ] *(opcional 8.5)* `scripts/signals.py` gerando o contrato `date→weight`
- [ ] *(opcional 8.5)* `scripts/backtest_pro.py` gerando `results/tearsheet.html` (quantstats), sem importar FinRL/FinRL-X
- [ ] *(opcional 8.5)* backtest com custos/turnover e `--to-daily` para anualização correta
- [ ] *(8.6)* `config.py` com splits train/val/trade (5 min) e `INDICATORS` diário+semanal congelados
- [ ] *(8.6)* adaptador emitindo train/val/trade a partir dos 7 CSVs concatenados
- [ ] *(8.6)* `scripts/run_experiment.sh` autônomo: dataset→calibração→treino(orçamento)→val→backtest, via `nohup` + checkpoints/resume
- [ ] *(8.6)* idle timeout do Codespace em 240 min antes do run longo
- [ ] *(8.6)* escolha do modelo pelo val; report final no trade via backtest 8.5

---

## Armadilhas conhecidas (para não perder tempo)

1. **`TA-Lib` na instalação:** use `pip install -e .` (pyproject), nunca `pip install -r requirements.txt`. O requirements inclui TA-Lib, que exige biblioteca C de sistema e quebra o build.
2. **Datas do BTC ≠ datas de ações:** o `config.py` vem com datas de mercado acionário. Ajuste para o alcance do seu CSV (com timestamp completo em intradiário), senão o `data_split` devolve DataFrames vazios.
3. **VIX/turbulence:** no adaptador, mantenha `use_vix=False` — VIX é do mercado de ações.
4. **CSV da CDD:** sempre `skiprows=1` (1ª linha é comentário) e reordene por data ascendente (vem decrescente).
5. **Timestamp completo (minuto):** mantenha `date` como `YYYY-MM-DD HH:MM:SS`. Se cortar só a data, várias barras do mesmo dia colidem e o pipeline quebra.
6. **Janelas de indicadores em minutos:** `rsi_30`/`close_30_sma` = 30 *barras* = 30 minutos (não 30 dias). Ajuste os números em `INDICATORS` se quiser janelas equivalentes maiores.
7. **`total_timesteps` de treino:** os exemplos assumem dados diários. Com milhões de barras, comece com timesteps baixos, valide, e só então aumente — senão o treino não termina.
8. **Memória:** ~4M linhas + indicadores estouram RAM baixa. Desenvolva reamostrado; para 1min puro, use VPS/Codespace com bastante RAM (16 GB+).
9. **Nunca versione nem "buildar" o CSV grande:** `.gitignore` + `.dockerignore` com `data/`. Ele vai à VPS por `scp` e é montado como volume.
10. **Cota do Codespaces:** pare o Codespace quando não usar.
11. **Imagem grande:** o FinRL puxa `ray`, `wrds`, `selenium` etc. (>2 GB). Normal. Dá para enxugar depois com um `pyproject` mínimo (stable-baselines3, stockstats, pandas) — otimização, não bloqueio.
12. **Nada disso é conselho financeiro.** É um pipeline de pesquisa/educação; resultados de backtest não se traduzem em lucro real.