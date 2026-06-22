# Monitoramento de violência — Trabalho Final SSL

Sistema de alerta em tempo quase real: captura por **câmera e microfone**, detecção de **grito**, **impacto sonoro** e **pedido de socorro** (em desenvolvimento), com envio automático à **central** (imagem + áudio do momento).

**Tema:** apoio à segurança e enfrentamento da violência de gênero (contexto acadêmico; triagem, não identificação criminal).

## O que você precisa para começar

| Requisito | Detalhe |
|-----------|---------|
| Python | 3.9+ (testado no 3.9 do macOS) |
| Webcam | Permissão de câmera no macOS |
| Microfone | Para detecção de grito/impacto/socorro |
| SO testado | macOS / Linux (Windows: ajustar índice da câmera) |

## Instalação rápida

```bash
cd TrabalhoFinal
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/setup_vosk.sh   # modelo de transcrição em português (~50 MB)
cp .env.example .env        # opcional — ajustar limiares
```

### Machine learning (recomendado para melhor acurácia)

```bash
# Download automático do NIGENS (Zenodo, ~2 GB) + preparar + treinar
python -m scripts.download_nigens --all --epochs 25
```

Guia completo: [docs/ml.md](docs/ml.md)

### O que **não** sobe para o Git

Por licença (NIGENS CC BY-NC-ND) e tamanho (~2 GB), **áudios e modelos ficam só na máquina local**:

| Conteúdo | Onde fica | Como obter |
|----------|-----------|------------|
| Dataset NIGENS (ZIP + WAV) | `data/datasets/nigens/` | `python -m scripts.download_nigens` |
| Clipes preparados | `data/datasets/prepared/` | `--prepare` (gerado automaticamente) |
| Modelo CNN treinado | `data/models/sound_classifier.pt` | `--train` ou `train_sound_model` |
| Modelo Vosk (PT) | `data/models/vosk-model-small-pt-0.3/` | `bash scripts/setup_vosk.sh` |

Quem clonar o repositório deve rodar os comandos acima antes de usar o cliente com ML/transcrição.

## Executar (dois terminais)

**Terminal 1 — Central:**

```bash
source .venv/bin/activate
uvicorn central.app:app --reload --host 127.0.0.1 --port 8000
```

- Painel operador: http://127.0.0.1:8000/central  
- API: http://127.0.0.1:8000/docs  

**Terminal 2 — Cliente (câmera + microfone):**

```bash
source .venv/bin/activate
python -m client.main
```

Na primeira execução no macOS, conceda permissão de **Câmera** e **Microfone** ao Terminal ou ao Python.

## Estrutura do repositório

```text
TrabalhoFinal/
├── client/           # Captura, detectores, ML, envio de alertas
├── scripts/          # prepare_nigens, train_sound_model
├── central/          # API + armazenamento + dashboard
├── shared/           # Modelos e configuração
├── data/
│   ├── alerts/       # Alertas recebidos (gerado em runtime, não versionado)
│   ├── datasets/     # NIGENS + prepared (baixar localmente — ver docs/ml.md)
│   ├── models/       # Vosk + CNN (gerar localmente)
│   └── samples/      # Vídeos/áudios de teste opcionais (não versionados)
├── docs/
│   ├── arquitetura.md
│   └── referencias.md
├── requirements.txt
└── README.md
```

## Configuração

Variáveis em `.env` (veja `.env.example`):

- `CENTRAL_URL` — URL da API (padrão: `http://127.0.0.1:8000`)
- `CAMERA_ID` — índice da webcam (0 = padrão)
- `AUDIO_CHUNK_SECONDS` — janela de áudio analisada (padrão: **5 s**)
- `SCREAM_ENERGY_THRESHOLD` / `IMPACT_ENERGY_THRESHOLD` — calibrar com testes
- `TRANSCRIPTION_ENABLED` / `VOSK_MODEL_PATH` — transcrição offline (Vosk)
- `HELP_KEYWORDS` — palavras que disparam alerta de socorro (após transcrever)

## Divisão sugerida (grupo de 4)

| Pessoa | Foco |
|--------|------|
| 1 | Detectores de áudio (grito, impacto) + calibração |
| 2 | Pedido de socorro (ASR / palavra-chave) |
| 3 | Central (API, storage, dashboard) |
| 4 | Documentação, testes, vídeo demo, ética/LGPD |

## Documentação do trabalho

- [Arquitetura](docs/arquitetura.md)
- [Referências](docs/referencias.md) — atualizar com todos os artigos citados

## Status do código

| Módulo | Status |
|--------|--------|
| Captura câmera + áudio | Funcional |
| Detecção grito / impacto | Heurística + **CNN (mel-spectrogram)** |
| Pedido de socorro | Transcrição Vosk + palavras-chave |
| Central + alertas | Funcional |
| Agressão em vídeo | Planejado |

## Licença e uso ético

Use apenas em ambiente de teste com consentimento dos participantes. Não grave nem armazene dados de terceiros sem base legal (LGPD).
