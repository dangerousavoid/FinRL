# CLAUDE.md — Projeto FinRL + Bitcoin

## Contexto
Fork do FinRL adaptado para operar SOMENTE com Bitcoin, dados de 1 minuto da
CryptoDataDownload. O plano completo está em `docs/plano_finrl_btc.md` — leia a fase
relevante antes de implementar.

## Regras inegociáveis
- NÃO refatorar o core do FinRL (pasta `finrl/`). É dependência upstream.
- Única edição permitida em arquivo existente: parâmetros hard-coded em `finrl/config.py`
  (datas de treino/trade e lista INDICATORS).
- Todo código novo vai em `scripts/`. Nunca espalhar lógica dentro de `finrl/`.
- Contrato de dados: colunas exatamente `date, open, high, low, close, volume, tic`.
  Em intradiário, `date` = timestamp completo `YYYY-MM-DD HH:MM:SS`.
- Cripto não tem VIX nem fundamentos: sempre `use_vix=False`, `use_turbulence=False`
  no FeatureEngineer e `turbulence_threshold=None` ao construir o env.
- Instalar com `pip install -e .`. NUNCA `pip install -r requirements.txt`.
- Um ativo só: `tic = "BTCUSDT"`.

## Dados
- Brutos da CDD (1 min, 2020–2026) em `data/raw/` (GITIGNORED).
- Cada CSV tem 1ª linha de comentário (skiprows=1) e vem em ordem decrescente (reordenar).
- São 7 arquivos por ano: o adaptador deve concatenar, deduplicar e ordenar todos.

## Validação (rodar após cada mudança)
- `python -c "import finrl; print('ok')"`
- Adaptador gera `train_data.csv` e `trade_data.csv` no período esperado.

## Workflow
- Uma fase por vez; um branch por fase; commit ao concluir.
- Use Plan Mode: proponha o plano e aguarde aprovação antes de editar.
- Comece pequeno: valide com `--resample 1h` antes do 1 minuto puro.

## Fora de escopo agora
- Fases 6 e 7 (VPS/deploy) NÃO serão feitas. Apenas ambiente de dev.