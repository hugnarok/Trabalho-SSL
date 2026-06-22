# Calibração — reduzir falsos positivos

## Como o sistema decide enviar alerta

1. Cada `DETECTION_INTERVAL_SECONDS` (ex.: 0,5 s) analisa os últimos `AUDIO_CHUNK_SECONDS` (5 s) de áudio.
2. Os detectores (`scream`, `impact`, `help_request`) retornam candidato.
3. Só dispara alerta se o **mesmo tipo** aparecer **`DETECTION_CONFIRMATIONS_REQUIRED` vezes seguidas** (ex.: 2 → ~1 s de confirmação).
4. A confiança deve ser ≥ `MIN_ALERT_CONFIDENCE`.

Na tela você verá `grito? (1/2)` antes de confirmar; após confirmar, vira `grito` e envia à central.

## Ajuste no `.env`

| Variável | Muitos falsos positivos | Não detecta quando deveria |
|----------|-------------------------|----------------------------|
| `SCREAM_ENERGY_THRESHOLD` | **Aumentar** (ex.: 0.03) | **Diminuir** (ex.: 0.02) |
| `SCREAM_HIGH_BAND_RATIO` | **Aumentar** (ex.: 1.5) | **Diminuir** (ex.: 1.2) |
| `IMPACT_ENERGY_THRESHOLD` | **Aumentar** | **Diminuir** |
| `IMPACT_PEAK_RATIO` | **Aumentar** (ex.: 6) | **Diminuir** (ex.: 4) |
| `MIN_ALERT_CONFIDENCE` | **Aumentar** (ex.: 0.65) | **Diminuir** (ex.: 0.45) |
| `DETECTION_CONFIRMATIONS_REQUIRED` | **3** | **1** |
| `ALERT_COOLDOWN_SECONDS` | **Aumentar** (ex.: 20) | — |

## Pedido de socorro (não é o modelo CNN)

O classificador treinado no NIGENS reconhece **grito**, **impacto** e **normal** — não pedido de socorro.

Socorro usa **Vosk (transcrição)** + **palavras-chave** em `HELP_KEYWORDS`. Só dispara alerta se o texto transcrito contiver uma frase configurada (ex.: `socorro`, `me ajuda`, `preciso de ajuda`).

| Variável | Efeito |
|----------|--------|
| `HELP_KEYWORDS` | Lista separada por vírgula; evite `ajuda` sozinha (dispara em "ajudar") |
| `HELP_MIN_WORD_CONFIDENCE` | Confiança mínima da palavra no Vosk (ex.: 0.45) |
| `HELP_MIN_CONFIDENCE` | Confiança mínima do alerta de socorro (ex.: 0.85) |
| `HELP_CONFIRMATIONS_REQUIRED` | Quantas leituras seguidas com a mesma frase (ex.: 2) |

Teste sem câmera: fale claramente *"socorro"* ou *"me ajuda"* no microfone, ou use `python -m client.calibrate arquivo.wav`.

Se **qualquer voz** dispara **grito** (não socorro), suba `ML_CONFIDENCE_THRESHOLD` (ex.: 0.75) ou defina `ML_HEURISTIC_FALLBACK=false`.

## Testar com arquivo de áudio

```bash
source .venv/bin/activate
python -m client.calibrate data/samples/seu_audio.wav
```

O script imprime RMS, banda aguda, pico e razão de impacto **sem** abrir a câmera.

## Vídeo no alerta

| Variável | Descrição |
|----------|-----------|
| `ALERT_SAVE_VIDEO` | `true` = envia clip MP4 (padrão) |
| `ALERT_VIDEO_SECONDS` | Duração do clip (3 ou 5) |
| `VIDEO_BUFFER_SECONDS` | Deve ser ≥ duração do clip (ex.: 6) |
| `ALERT_SAVE_SNAPSHOT` | `true` se quiser JPEG **além** do vídeo |

## Dicas práticas

- Calibre em **silêncio** e **ruído ambiente** primeiro; anote os valores no terminal (`message` do alerta).
- Impactos de teclado/porta costumam disparar `impacto` — suba `IMPACT_PEAK_RATIO`.
- Música/TV pode disparar `grito` — suba `SCREAM_HIGH_BAND_RATIO`.
- Para demo estável: `DETECTION_CONFIRMATIONS_REQUIRED=3` e limiares um pouco mais altos que o ambiente real.
