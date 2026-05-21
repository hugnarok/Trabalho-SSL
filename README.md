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
├── client/           # Captura, detectores, envio de alertas
├── central/          # API + armazenamento + dashboard
├── shared/           # Modelos e configuração
├── data/
│   ├── alerts/       # Alertas recebidos (gerado em runtime)
│   └── samples/      # Vídeos/áudios de teste (vocês adicionam)
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
| Detecção grito / impacto | Heurística inicial (calibrar) |
| Pedido de socorro | Transcrição Vosk + palavras-chave |
| Central + alertas | Funcional |
| Agressão em vídeo | Planejado |

## Licença e uso ético

Use apenas em ambiente de teste com consentimento dos participantes. Não grave nem armazene dados de terceiros sem base legal (LGPD).
