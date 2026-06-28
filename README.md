# SafeAlert — Sistema de Monitoramento Acústico

Sistema distribuído de detecção de situações de risco em ambiente acadêmico. Opera por **câmera e microfone** e identifica três tipos de evento: **grito**, **impacto sonoro** e **pedido de socorro verbal**. Ao confirmar uma anomalia, envia automaticamente à central um clipe de vídeo retroativo, áudio e metadados do incidente.

> **Escopo ético:** ferramenta de triagem e registro de evidências para apoio operacional. Não realiza identificação de pessoas e não substitui perícia. Use apenas com consentimento dos participantes e base legal (LGPD).

---

## Índice

- [Como funciona](#como-funciona)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Executando o sistema](#executando-o-sistema)
- [Calibração dos limiares](#calibração-dos-limiares)
- [Machine Learning](#machine-learning)
- [Configuração completa (.env)](#configuração-completa-env)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Scripts utilitários](#scripts-utilitários)
- [Documentação técnica](#documentação-técnica)
- [Licença e uso ético](#licença-e-uso-ético)

---

## Como funciona

O sistema é composto por dois processos independentes que se comunicam via HTTP:

```
┌─────────────────────────────────┐        HTTP multipart        ┌──────────────────────────┐
│  Cliente (borda)                │ ────────────────────────────►│  Central (servidor)      │
│  Câmera + Microfone             │  meta.json + clip.mp4 + wav  │  FastAPI + Dashboard     │
│  CNN · Heurística RMS · Vosk   │                              │  data/alerts/<uuid>/     │
└─────────────────────────────────┘                              └──────────────────────────┘
```

**Cliente:** captura áudio em buffer circular e frames de vídeo contínuos. A cada 0,5 s, três detectores rodam em paralelo sobre a mesma janela de áudio:

| Detector | Método | Papel |
|---|---|---|
| CNN (mel-espectrograma) | Aprendizado de máquina (PyTorch) | Classificação semântica: grito / impacto / normal |
| Heurística de energia | Limiar RMS + razão de banda aguda | Fallback e redundância — não depende de modelo |
| ASR offline (Vosk) | Reconhecimento de fala em PT | Detecta palavras-chave: "socorro", "me ajuda" etc. |

Dois ciclos consecutivos com o mesmo tipo de evento confirmam um alerta. O cliente então extrai os **últimos 5 segundos de vídeo** do buffer (captura retroativa — o evento já aconteceu) e envia o pacote à central.

**Central:** recebe os alertas, persiste os arquivos em disco e os exibe no painel do operador com players de vídeo e áudio inline.

---

## Requisitos

| Item | Detalhe |
|---|---|
| Python | 3.9 ou superior (testado: 3.11 no Linux/macOS, 3.9 no macOS) |
| Webcam | Qualquer câmera USB ou integrada |
| Microfone | Qualquer entrada de áudio do sistema |
| RAM | Mínimo 2 GB livres (modelo Vosk + PyTorch) |
| Disco | ~3 GB para dataset NIGENS + modelos (apenas para treino) |
| SO testado | macOS · Linux (Ubuntu 22+) · Windows (ajustar `CAMERA_ID`) |

---

## Instalação

### 1. Clonar e criar ambiente virtual

```bash
git clone <url-do-repositório>
cd Trabalho-SSL
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Copiar configuração

```bash
cp .env.example .env
```

Edite `.env` se necessário (câmera, URL da central, limiares). Os padrões funcionam para uso local.

### 3. Baixar o modelo de transcrição de voz (Vosk, ~50 MB)

```bash
bash scripts/setup_vosk.sh
```

O script baixa o modelo `vosk-model-small-pt-0.3` em `data/models/` automaticamente. Se já existir, não faz nada.

### 4. (Recomendado) Treinar o classificador acústico CNN

Sem o modelo CNN, o sistema opera apenas com heurística de energia e ASR — o que ainda funciona, mas com menor riqueza de classificação.

```bash
# Baixa NIGENS (~2 GB), prepara os clipes e treina a CNN em uma só linha:
python -m scripts.download_nigens --all --epochs 25
```

Veja a seção [Machine Learning](#machine-learning) para mais detalhes e opções.

---

## Executando o sistema

Abra **dois terminais** com o ambiente virtual ativado:

**Terminal 1 — Central de alertas:**

```bash
uvicorn central.app:app --reload --host 127.0.0.1 --port 8000
```

| Endereço | O que é |
|---|---|
| http://127.0.0.1:8000/central | Painel do operador (alertas + vídeos) |
| http://127.0.0.1:8000/docs | Documentação interativa da API REST |

**Terminal 2 — Cliente (câmera + microfone):**

```bash
python -m client.main
```

Uma janela de preview abre com a imagem da câmera e a forma de onda de áudio em tempo real. A cor da onda muda de verde → amarelo → vermelho conforme o nível de risco detectado. Pressione **`q`** para encerrar.

> **macOS:** na primeira execução, o sistema pedirá permissão de câmera e microfone ao Terminal (ou ao Python). Conceda e reinicie o cliente se necessário.

> **Windows:** se a câmera não abrir, ajuste `CAMERA_ID=1` (ou 2) no `.env`.

---

## Calibração dos limiares

Os limiares da heurística de energia (`SCREAM_ENERGY_THRESHOLD`, `IMPACT_PEAK_RATIO` etc.) devem ser ajustados para o ambiente acústico específico onde o sistema será implantado.

Use o script de calibração com um arquivo WAV de referência:

```bash
# Teste com um arquivo de grito:
python -m client.calibrate data/datasets/prepared/grito/femaleScream_0000.wav

# Teste com um arquivo normal (não deve disparar):
python -m client.calibrate data/datasets/prepared/normal/footsteps_0000.wav

# Teste com um áudio próprio:
python -m client.calibrate meu_audio.wav
```

O script imprime os valores atuais dos limiares e indica se cada detector dispararia para aquele áudio, sem precisar de câmera nem conectar à central. Ajuste os valores no `.env` e re-execute até encontrar limiares adequados.

---

## Machine Learning

O classificador CNN converte janelas de áudio em mel-espectrogramas e os classifica em três categorias: **normal**, **grito** e **impacto**.

### Treino com NIGENS (recomendado)

```bash
# Tudo de uma vez: download + preparação + treino
python -m scripts.download_nigens --all --epochs 25

# Ou etapas separadas:
python -m scripts.download_nigens           # só baixa e extrai (~2 GB)
python -m scripts.download_nigens --prepare # organiza em grito/impacto/normal
python -m scripts.train_sound_model --epochs 25  # treina a CNN
```

O dataset NIGENS (licença CC BY-NC-ND 4.0) é baixado diretamente do Zenodo. **Não commite os arquivos de dataset ou modelo** — estão no `.gitignore`. Cada membro do grupo deve baixar e treinar localmente.

### Mapeamento NIGENS → classes do projeto

| Pastas do NIGENS | Classe |
|---|---|
| `femaleScream`, `maleScream` | `grito` |
| `crash`, `knock` | `impacto` |
| `femaleSpeech`, `maleSpeech`, `general`, `piano`, `footsteps`, `phone`, `engine`, `alarm`, `fire`, `dog`, `baby` | `normal` |

### Treino com áudios próprios

Coloque arquivos `.wav` em:

```
data/samples/ml/grito/
data/samples/ml/impacto/
data/samples/ml/normal/
```

E execute `python -m scripts.train_sound_model --epochs 25` normalmente. Os dois diretórios (`datasets/prepared/` e `samples/ml/`) são combinados automaticamente.

### Avaliação do modelo

```bash
python -m scripts.evaluate_model
```

Imprime acurácia, F1-Score por classe e exibe a matriz de confusão. Útil para verificar se o modelo treinado está generalizando antes de colocá-lo em produção.

### Artefatos gerados

| Arquivo | Descrição |
|---|---|
| `data/models/sound_classifier.pt` | Pesos da CNN (PyTorch) |
| `data/models/sound_classifier.json` | Metadados: classes, taxa de amostragem, métricas |

### Variáveis de configuração ML

| Variável | Padrão | Descrição |
|---|---|---|
| `ML_ENABLED` | `true` | Ativa o classificador CNN se o modelo existir |
| `ML_CONFIDENCE_THRESHOLD` | `0.65` | Probabilidade mínima para disparar alerta via CNN |
| `ML_HEURISTIC_FALLBACK` | `true` | Mantém heurística RMS rodando em paralelo com a CNN |

Defina `ML_HEURISTIC_FALLBACK=false` para usar apenas a CNN (não recomendado sem modelo bem calibrado).

---

## Configuração completa (.env)

| Variável | Padrão | Descrição |
|---|---|---|
| `CENTRAL_URL` | `http://127.0.0.1:8000` | URL da API da central |
| `CAMERA_ID` | `0` | Índice da webcam (0 = padrão do sistema) |
| `CAMERA_LABEL` | `cam-01` | Nome identificador da câmera nos alertas |
| `SAMPLE_RATE` | `16000` | Taxa de amostragem do áudio em Hz |
| `AUDIO_CHUNK_SECONDS` | `5.0` | Duração da janela de áudio analisada por ciclo |
| `DETECTION_INTERVAL_SECONDS` | `0.5` | Intervalo entre ciclos de detecção |
| `DISPLAY_FPS` | `30` | FPS da janela de preview local |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | `640` / `480` | Resolução da captura de vídeo |
| `ALERT_COOLDOWN_SECONDS` | `15.0` | Intervalo mínimo entre alertas consecutivos |
| `DETECTION_CONFIRMATIONS_REQUIRED` | `2` | Ciclos consecutivos necessários para confirmar alerta |
| `MIN_ALERT_CONFIDENCE` | `0.55` | Confiança mínima global para qualquer alerta |
| `SCREAM_ENERGY_THRESHOLD` | `0.025` | Limiar de energia RMS para detector de grito |
| `SCREAM_HIGH_BAND_RATIO` | `1.35` | Razão mínima de energia na banda 2–8 kHz para grito |
| `IMPACT_ENERGY_THRESHOLD` | `0.018` | Limiar RMS para detector de impacto |
| `IMPACT_PEAK_RATIO` | `5.0` | Pico mínimo relativo à mediana para impacto |
| `ALERT_SAVE_VIDEO` | `true` | Salva clipe MP4 retroativo de 5 s junto ao alerta |
| `ALERT_VIDEO_SECONDS` | `5.0` | Duração do clipe de evidência |
| `VIDEO_BUFFER_SECONDS` | `6.0` | Tamanho do buffer circular de vídeo |
| `ALERT_SAVE_SNAPSHOT` | `false` | Salva frame JPEG (alternativa ao vídeo) |
| `TRANSCRIPTION_ENABLED` | `true` | Ativa detector de palavras-chave via Vosk |
| `VOSK_MODEL_PATH` | `data/models/vosk-model-small-pt-0.3` | Caminho do modelo Vosk |
| `HELP_KEYWORDS` | `socorro,me ajuda,...` | Palavras/frases que disparam alerta de socorro |
| `HELP_MIN_CONFIDENCE` | `0.85` | Confiança mínima da correspondência de palavra-chave |
| `HELP_MIN_WORD_CONFIDENCE` | `0.45` | Confiança mínima por palavra na transcrição Vosk |
| `HELP_CONFIRMATIONS_REQUIRED` | `2` | Confirmações necessárias para alerta de socorro |
| `ML_ENABLED` | `true` | Ativa o classificador CNN |
| `ML_CONFIDENCE_THRESHOLD` | `0.65` | Limiar de probabilidade da CNN |
| `ML_HEURISTIC_FALLBACK` | `true` | Mantém heurística em paralelo com a CNN |

---

## Estrutura do repositório

```
Trabalho-SSL/
│
├── client/                     # Módulo cliente (borda)
│   ├── main.py                 # Ponto de entrada: loop principal + preview
│   ├── capture.py              # Buffer circular de áudio e vídeo (StreamCapture)
│   ├── detection_worker.py     # Thread de detecção paralela + streak counter
│   ├── alert_sender.py         # Empacotamento e envio multipart à central
│   ├── transcription.py        # Interface com o Vosk (ASR offline)
│   ├── calibrate.py            # Ferramenta de calibração de limiares via WAV
│   ├── waveform_overlay.py     # Visualização da forma de onda no preview
│   ├── detectors/
│   │   ├── base.py             # Dataclass DetectionResult
│   │   ├── scream.py           # Heurística: energia RMS + banda aguda (2–8 kHz)
│   │   ├── impact.py           # Heurística: pico de energia relativo à mediana
│   │   ├── help_request.py     # Palavras-chave via Vosk
│   │   └── ml_events.py        # Interface com o classificador CNN
│   └── ml/
│       ├── model_arch.py       # Definição da arquitetura CNN (PyTorch)
│       ├── features.py         # Extração de mel-espectrograma (STFT + banco mel)
│       ├── classifier.py       # Singleton: carregamento e inferência do modelo
│       └── audio_io.py         # Leitura e reamostragem de arquivos WAV
│
├── central/                    # Módulo servidor
│   ├── app.py                  # API REST FastAPI + painel HTML do operador
│   ├── storage.py              # Persistência de alertas em data/alerts/<uuid>/
│   ├── media.py                # Utilitários de leitura de mídia
│   └── video_utils.py          # Codificação MP4 com faststart
│
├── shared/                     # Código compartilhado
│   ├── config.py               # Pydantic Settings — lê variáveis do .env
│   └── models.py               # Tipos: EventType, DetectionResult, AlertResponse
│
├── scripts/                    # Ferramentas de treino e manutenção
│   ├── download_nigens.py      # Download + extração + preparação do NIGENS
│   ├── prepare_nigens.py       # Organiza clipes NIGENS por classe
│   ├── train_sound_model.py    # Treina a CNN e salva os pesos
│   ├── evaluate_model.py       # Avalia o modelo: F1, acurácia, matriz de confusão
│   ├── fix_alert_videos.py     # Corrige MP4s antigos sem flag faststart
│   ├── setup_vosk.sh           # Baixa o modelo Vosk PT (~50 MB)
│   └── zenodo_download.py      # Helper de download do Zenodo
│
├── docs/
│   ├── arquitetura.md          # Descrição detalhada da arquitetura e fluxo
│   └── ml.md                   # Guia completo do pipeline de ML
│
├── data/                       # Gerado em runtime — não versionado
│   ├── alerts/                 # Alertas recebidos pela central
│   ├── datasets/               # NIGENS + clipes preparados (baixar localmente)
│   ├── models/                 # Modelo CNN + Vosk (gerar localmente)
│   └── samples/                # Áudios e vídeos de teste opcionais
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Scripts utilitários

| Script | Como usar | O que faz |
|---|---|---|
| `scripts/setup_vosk.sh` | `bash scripts/setup_vosk.sh` | Baixa modelo Vosk PT em `data/models/` |
| `scripts/download_nigens.py` | `python -m scripts.download_nigens --all` | Download + preparo + treino do NIGENS |
| `scripts/train_sound_model.py` | `python -m scripts.train_sound_model --epochs 25` | Treina a CNN com dados em `data/datasets/prepared/` e `data/samples/ml/` |
| `scripts/evaluate_model.py` | `python -m scripts.evaluate_model` | Avalia o modelo: F1 por classe, matriz de confusão |
| `client/calibrate.py` | `python -m client.calibrate audio.wav` | Testa os detectores num arquivo WAV sem câmera |
| `scripts/fix_alert_videos.py` | `python -m scripts.fix_alert_videos` | Corrige MP4s antigos que não reproduzem no navegador |

---

## Documentação técnica

- [Arquitetura do sistema](docs/arquitetura.md) — fluxo de dados, componentes, decisões de design
- [Pipeline de Machine Learning](docs/ml.md) — mel-espectrograma, treinamento, avaliação

---

## Licença e uso ético

O código-fonte deste projeto é de uso acadêmico. O dataset NIGENS é distribuído sob licença **CC BY-NC-ND 4.0** (uso não comercial, sem modificação, com atribuição). Os modelos treinados derivam do NIGENS e estão sujeitos à mesma licença.

**Diretrizes de uso:**

- Execute o sistema apenas em ambiente de teste com consentimento explícito dos participantes.
- Não armazene dados de terceiros sem base legal conforme a **Lei nº 13.709/2018 (LGPD)**.
- O sistema não realiza identificação de pessoas e não deve ser usado como ferramenta de vigilância irrestrita.
- Para implantação real em campus universitário, consulte a assessoria jurídica da instituição.
