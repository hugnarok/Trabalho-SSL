# Arquitetura do Sistema — SafeAlert

> **Escopo acadêmico:** ferramenta de triagem e registro de evidências para apoio operacional em ambiente controlado. Não substitui perícia nem realiza identificação judicial de autores.

---

## Visão Geral

O SafeAlert adota uma arquitetura **cliente-servidor assíncrona** com dois processos independentes:

```
┌────────────────────────────────────────────┐
│  CLIENTE (borda)                           │
│                                            │
│  Câmera → Buffer circular de frames        │
│  Microfone → AudioRingBuffer (12 s)        │
│                                            │
│  DetectionWorker (thread dedicada)         │
│  ├── CNNClassifier   ─┐                    │
│  ├── Heurística RMS  ─┼─ fusão → streak   │
│  └── Vosk ASR        ─┘  counter (k=2)    │
│                  │                         │
│            alerta confirmado               │
│                  │                         │
│  AlertSender: WAV + MP4 (retroativo) + JSON│
└────────────────────┬───────────────────────┘
                     │ HTTP POST multipart/form-data
                     ▼
┌────────────────────────────────────────────┐
│  CENTRAL (servidor)                        │
│                                            │
│  FastAPI (Uvicorn ASGI)                    │
│  POST /api/alerts → storage.py             │
│  GET  /central    → Dashboard HTML         │
│                                            │
│  data/alerts/<uuid>/                       │
│  ├── meta.json                             │
│  ├── clip.mp4  (vídeo retroativo 5 s)      │
│  └── clip.wav  (áudio do evento)           │
└────────────────────────────────────────────┘
```

Os dois módulos se comunicam exclusivamente via API REST. O cliente não tem acesso direto ao banco de dados nem ao sistema de arquivos da central, e a central não controla a captura do cliente — o acoplamento é mínimo e intencional.

---

## Módulo Cliente

### Captura Multimodal (`capture.py` — `StreamCapture`)

A captura de áudio e vídeo ocorre em **threads independentes**, nunca bloqueando o laço principal.

**Áudio — `AudioRingBuffer`:**

- Buffer circular de tamanho fixo (padrão: 12 s × 16.000 Hz = 192.000 amostras).
- Alimentado pelo _callback_ assíncrono do `sounddevice`, chamado em thread de I/O de áudio.
- Escrita protegida por `threading.Lock` para acesso seguro da thread de detecção.
- Leitura retorna uma cópia do segmento mais recente sem pausar a captura.
- Sem alocações dinâmicas em runtime: o ponteiro de escrita avança módulo o tamanho fixo, sobrescrevendo amostras antigas.

**Vídeo — `_VideoReaderThread`:**

- Loop contínuo de leitura de frames via OpenCV (`cv2.VideoCapture`).
- Frames armazenados em `collections.deque(maxlen=N)` — quando cheio, o frame mais antigo é descartado automaticamente.
- `maxlen` é calculado como `VIDEO_BUFFER_SECONDS × CAMERA_FPS` (padrão: 6 s × 30 FPS = 180 frames).
- **Captura retroativa:** ao confirmar um alerta, os últimos N frames do deque são extraídos. O clipe de evidência contém os segundos _anteriores_ ao disparo, não os posteriores — ou seja, captura o evento que motivou o alerta.
- Codificação em H.264/MP4 via `imageio-ffmpeg` com flag `faststart` (átomo `moov` no início do arquivo), permitindo reprodução imediata no navegador sem download completo. Fallback para codec `mp4v` do OpenCV quando FFmpeg não está disponível.

### Pipeline de Detecção Paralela (`detection_worker.py` — `DetectionWorker`)

Opera como thread dedicada, executando o ciclo de detecção a cada `DETECTION_INTERVAL_SECONDS` (padrão: 0,5 s).

```
Janela de áudio (5 s)
        │
        ├──► CNNClassifier.predict()     → DetectionResult(grito | impacto | normal, conf)
        ├──► detect_scream()             → DetectionResult(grito, conf)
        ├──► detect_impact()             → DetectionResult(impacto, conf)
        └──► detect_help_request()       → DetectionResult(socorro, conf)
                    │
              fusão: candidato = max(detected, key=confidence)
                    │
              _update_streak(candidato)
                    │
              se streak >= k:
                    └──► state.best = candidato  →  disparo de alerta
```

**Mecanismo de streak (anti-ruído):**

- Mantém um contador de ciclos consecutivos com o mesmo tipo de evento.
- O alerta só é promovido a `state.best` após `DETECTION_CONFIRMATIONS_REQUIRED` confirmações consecutivas (padrão: k=2).
- Se o tipo de evento muda entre ciclos, o contador reinicia.
- Custo: adiciona `k × DETECTION_INTERVAL_SECONDS` de latência ao disparo (padrão: +1 s).
- Benefício: elimina disparos por eventos acústicos transitórios isolados (tosse, batida de porta única).

**Fusão de detectores:**

Quando múltiplos detectores disparam no mesmo ciclo, o candidato com maior `confidence` é selecionado como representante. Isso significa que heurística e CNN competem — se a CNN identificar grito com 0,87 e a heurística identificar impacto com 0,91, o candidato do ciclo será impacto/heurística.

### Detectores

#### Heurística de Grito (`detectors/scream.py`)

Combina dois critérios calculados sobre a janela de áudio completa:

1. **Energia RMS:** `E_RMS ≥ SCREAM_ENERGY_THRESHOLD` — indica que o evento tem volume significativo.
2. **Razão de banda aguda:** energia na faixa 2–8 kHz (calculada via STFT) dividida pela energia total. Gritos concentram energia em frequências agudas. Se a razão supera `SCREAM_HIGH_BAND_RATIO`, o critério é satisfeito.

Disparo: **ambos** os critérios devem ser satisfeitos (AND lógico). O `score` final é uma média ponderada 50/50 das duas razões normalizadas pelos limiares.

#### Heurística de Impacto (`detectors/impact.py`)

Analisa a distribuição de energia em janelas curtas de 50 ms sobre o segmento de áudio:

- Calcula o **pico** de energia e a **mediana** das energias por janela.
- Um pico superior a `IMPACT_PEAK_RATIO × mediana` (padrão: 5×) é classificado como impacto.
- A lógica captura eventos de início abrupto: batidas, quedas, golpes — caracterizados por pico de energia muito acima do nível médio do ambiente.

#### Detector de Socorro (`detectors/help_request.py`)

Opera em background via Vosk (ASR offline):

- Transcreve o áudio continuamente sem enviar dados para nenhum servidor externo.
- Compara o texto transcrito com `HELP_KEYWORDS` usando expressões regulares com delimitadores de palavra (`\b`), evitando que "ajuda" seja detectado dentro de "ajudar".
- Filtra palavras com confiança abaixo de `HELP_MIN_WORD_CONFIDENCE` (padrão: 0,45) antes de comparar.
- Frases compostas (ex: "me ajuda") têm prioridade sobre palavras isoladas.

#### Classificador CNN (`detectors/ml_events.py` + `ml/`)

- Carregado como singleton com lazy loading ao primeiro uso.
- Extrai mel-espectrograma da janela de áudio: STFT com janela de Hann (`N_FFT=512`, `Hop=256`), banco de 64 filtros mel, log-compressão, normalização z-score.
- Resultado: tensor `1 × 64 × 128` passado à CNN.
- A predição retorna probabilidades softmax para três classes. Se a probabilidade da classe de maior confiança supera `ML_CONFIDENCE_THRESHOLD` e a classe não é `normal`, gera um `DetectionResult`.
- Em caso de erro de carregamento (modelo não encontrado, versão incompatível), falha silenciosamente — a heurística de energia assume o papel de fallback.

### Envio de Alertas (`alert_sender.py`)

Ao confirmar um alerta (`state.best` não nulo e cooldown expirado), o módulo `AlertSender`:

1. Captura o clipe MP4 retroativo do buffer de vídeo.
2. Converte o segmento de áudio para WAV em memória.
3. Monta payload `multipart/form-data` com: `meta` (JSON com tipo, confiança, timestamp UTC, label da câmera, transcrição), `video` (bytes MP4), `audio` (bytes WAV).
4. Executa o POST em thread separada (`daemon=True`) para não bloquear o laço de captura.
5. Após envio bem-sucedido, registra o `alert_id` retornado pela central e atualiza o timestamp de cooldown.

---

## Módulo Central

### API REST (`central/app.py`)

Construída sobre FastAPI com Uvicorn (ASGI), suporta múltiplas requisições simultâneas sem bloqueio de I/O.

| Endpoint | Método | Descrição |
|---|---|---|
| `/api/alerts` | POST | Recebe alerta multipart; salva em disco; retorna `alert_id` |
| `/api/alerts` | GET | Lista alertas em ordem cronológica (mais recente primeiro) |
| `/api/alerts/{id}` | GET | Retorna metadados de um alerta específico |
| `/api/alerts/{id}/{media}` | GET | Retorna arquivo de mídia (`clip.mp4`, `clip.wav`, `snapshot.jpg`) |
| `/api/alerts` | DELETE | Remove todos os alertas (para desenvolvimento/testes) |
| `/central` | GET | Painel HTML do operador |
| `/docs` | GET | Documentação interativa OpenAPI |

A rota de mídia valida o caminho contra _path traversal_: o caminho resolvido deve permanecer dentro de `data/alerts/`.

### Persistência (`central/storage.py`)

Cada alerta é salvo numa pasta `data/alerts/<uuid>/` contendo:

```
data/alerts/
└── 3f7a2c1e-…/
    ├── meta.json      # tipo, confiança, timestamp, câmera, transcrição
    ├── clip.mp4       # clipe de vídeo retroativo (H.264, faststart)
    └── clip.wav       # áudio do evento (PCM 16 kHz)
```

O UUID é gerado por `uuid.uuid4()` no momento do recebimento, garantindo unicidade sem coordenação entre clientes. A listagem de alertas lê os `meta.json` de todas as pastas e ordena por `received_at`.

### Dashboard do Operador

Painel HTML estático servido em `/central` com atualização automática a cada 5 s via _polling_ assíncrono. Exibe para cada alerta: tipo de evento, nível de confiança, timestamp, câmera de origem, transcrição (se disponível), player de vídeo inline e player de áudio inline.

---

## Configuração e Modularidade

Toda a configuração é lida de variáveis de ambiente (arquivo `.env`) via `pydantic-settings` (`shared/config.py`). Isso garante que:

- Nenhum valor está "hardcoded" no código de produção.
- Diferentes ambientes (laboratório A, laboratório B, produção) usam arquivos `.env` distintos sem modificar o código.
- A documentação dos parâmetros está centralizada em `.env.example`.

Os tipos e modelos compartilhados entre cliente e central (`EventType`, `DetectionResult`, `AlertResponse`) vivem em `shared/models.py`, evitando duplicação e garantindo consistência de contrato entre os dois processos.

---

## Decisões de Design

| Decisão | Justificativa |
|---|---|
| Buffer circular retroativo de vídeo | Captura o evento que causou o alerta, não o período após o disparo |
| Três detectores em paralelo | Nenhum detector é suficiente isoladamente; cada um cobre lacunas dos outros |
| Streak counter (k=2) | Reduz falsos positivos de eventos transitórios sem adicionar latência excessiva |
| ASR offline (Vosk) | Elimina transmissão de voz para servidores externos — requisito LGPD |
| Heurística como fallback da CNN | CNN tem F1 baixo para impacto (0,13); heurística garante detecção de picos abruptos |
| Clipe de 5 s máximo | Minimiza armazenamento e enquadra o sistema como triagem, não vigilância contínua |
| FastAPI + Uvicorn | I/O assíncrono nativo permite múltiplos clientes simultâneos sem bloqueio |

---

## Fluxo Completo de um Alerta

```
t=0,0 s  Evento acústico ocorre (ex: grito)
t=0,5 s  DetectionWorker processa janela → candidato: grito, conf=0.88 (streak=1)
t=1,0 s  DetectionWorker confirma → grito, conf=0.91 (streak=2 ≥ k=2)
          → state.best = DetectionResult(SCREAM, 0.91)
t=1,0 s  Loop principal detecta state.best → dispara _send_alert_async em thread
t=1,0 s  AlertSender extrai últimos 5 s do buffer de vídeo (frames de t=-5 s a t=0 s)
t=1,0 s  AlertSender converte 5 s de áudio → WAV; monta multipart
t=1,1 s  POST /api/alerts → Central recebe, salva em data/alerts/<uuid>/
t=1,1 s  Operador vê alerta no dashboard (próxima atualização em até 5 s)
```

Latência total típica do evento acústico até disponibilidade no dashboard: **6–7 segundos** (dominada pelo buffer de áudio de 5 s + confirmação de 1 s).

---

## Próximos Passos

- [ ] Adicionar `class_weight` no `CrossEntropyLoss` do treinamento para melhorar F1 nas classes minoritárias
- [ ] Implementar autenticação por API key nas rotas administrativas (`DELETE`, `POST`)
- [ ] Substituir ordenação de alertas por UUID por ordenação por `received_at`
- [ ] Adicionar retry com backoff exponencial no `AlertSender`
- [ ] Integrar estimação de pose humana (OpenPose / MoveNet) para fusão multimodal real
- [ ] Substituir persistência em sistema de arquivos por banco de dados (SQLite ou PostgreSQL) para escalabilidade
