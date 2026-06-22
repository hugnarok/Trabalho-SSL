# Machine Learning — classificação de sons

## Arquitetura

```text
Áudio (5 s, 16 kHz)
  → log-mel spectrogram (STFT + banco mel)     [SSL]
  → CNN leve (PyTorch)
  → classes: normal | grito | impacto
  → confirmação em sequência + alerta central
```

**Socorro** continua por **Vosk** (fala), não pelo CNN.

## Passo a passo

### 1. Dependências

```bash
pip install torch
```

### 2. Dataset NIGENS — download automático (recomendado)

```bash
# Download (~2 GB) + extrair + preparar + treinar (tudo de uma vez)
python -m scripts.download_nigens --all --epochs 25
```

Etapas separadas:

```bash
python -m scripts.download_nigens              # só baixa e extrai
python -m scripts.download_nigens --prepare    # organiza grito/impacto/normal
python -m scripts.train_sound_model            # treina CNN
```

Fonte: https://zenodo.org/records/2535878 (licença CC BY-NC-ND 4.0 — uso acadêmico).

**Não commite** `data/datasets/` nem `data/models/` — estão no `.gitignore`. Cada integrante do grupo baixa/treina localmente.

### Mapeamento NIGENS → classes do projeto

| Pasta NIGENS | Classe de treino |
|--------------|------------------|
| `femaleScream`, `maleScream` | grito |
| `crash`, `knock` | impacto |
| `general`, `piano`, `footsteps`, `phone`, `engine`, `alarm` | normal |
| `femaleSpeech`, `maleSpeech` | normal (fala cotidiana) |
| `fire`, `dog`, `baby` | normal (sons ambiente — reduz falso positivo) |

Após `--prepare`, o script lista quantos arquivos foram usados e se alguma pasta ficou de fora.

### Download manual (se o automático falhar)

1. Baixe `NIGENS.zip` em https://zenodo.org/records/2535878  
2. Coloque em `data/datasets/nigens/downloads/NIGENS.zip`  
3. Rode:

```bash
python -m scripts.download_nigens --skip-download --all --epochs 25
```

No macOS a extração usa `unzip` do sistema (ZIP com Deflate64).

### 3. Ou use gravações próprias

Coloque WAVs em:

```text
data/samples/ml/grito/
data/samples/ml/impacto/
data/samples/ml/normal/
```

### 4. Treinar

```bash
python -m scripts.train_sound_model --epochs 25
```

Saída:

- `data/models/sound_classifier.pt`
- `data/models/sound_classifier.json`

### 5. Rodar o cliente

```bash
python -m client.main
```

No terminal: mensagens `ML: grito (...)` quando o modelo dispara.

## Configuração (`.env`)

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `ML_ENABLED` | true | Usa CNN se o modelo existir |
| `ML_CONFIDENCE_THRESHOLD` | 0.65 | Limiar de probabilidade |
| `ML_HEURISTIC_FALLBACK` | true | Mantém detectores antigos se ML ausente ou em paralelo |

Para **só ML** (após calibrar): `ML_HEURISTIC_FALLBACK=false`

## Artigo IEEE

Compare:

1. Heurística (RMS + FFT)
2. CNN + mel-spectrogram
3. Métricas: accuracy, precision, recall, F1 no conjunto de teste (impresso ao treinar)
