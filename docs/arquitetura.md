# Arquitetura do sistema

## Visão geral

Sistema de monitoramento em tempo quase real para apoio à detecção de situações de violência (grito, impacto sonoro, pedido de socorro), com envio imediato de alertas à central operacional.

> **Escopo acadêmico:** ferramenta de triagem e registro de evidências em ambiente controlado. Não substitui perícia nem identificação judicial de autores.

## Componentes

```text
┌─────────────────┐         HTTP POST          ┌──────────────────┐
│  client/        │  (multipart: meta +        │  central/        │
│  - captura      │   snapshot.jpg + clip.wav)│  - FastAPI       │
│  - detectores   │ ─────────────────────────► │  - storage       │
│  - alert_sender │                            │  - dashboard     │
└─────────────────┘                            └──────────────────┘
```

### Cliente (`client/`)

| Módulo | Responsabilidade |
|--------|------------------|
| `capture.py` | Webcam (OpenCV) + microfone (sounddevice) |
| `detectors/scream.py` | Energia RMS + banda aguda (FFT) |
| `detectors/impact.py` | Pico de energia por frame |
| `detectors/help_request.py` | Palavra-chave (stub → ASR) |
| `alert_sender.py` | POST para `/api/alerts` |
| `main.py` | Loop principal + janela de preview |

### Central (`central/`)

| Módulo | Responsabilidade |
|--------|------------------|
| `app.py` | API REST + painel HTML em `/central` |
| `storage.py` | Persistência em `data/alerts/<uuid>/` |

### Compartilhado (`shared/`)

- `models.py` — tipos de evento e payload
- `config.py` — variáveis de ambiente (Pydantic Settings)

## Fluxo de um alerta

1. Cliente grava janela de áudio (padrão: 2 s) e lê um frame da câmera.
2. Detectores retornam `DetectionResult` (detectado, tipo, confiança).
3. Se houver detecção e cooldown expirado → monta JPEG + WAV → POST.
4. Central salva `meta.json`, `snapshot.jpg`, `clip.wav`.
5. Operador consulta `http://localhost:8000/central`.

## Próximos passos de desenvolvimento

- [ ] Calibrar limiares com vídeos em `data/samples/`
- [ ] Integrar ASR para "socorro" (Vosk ou Whisper tiny)
- [ ] Detector de agressão em vídeo (fluxo óptico / pose)
- [ ] Testes automatizados com clips conhecidos
- [ ] Autenticação na API (opcional)

## Conceitos de sinais (SSL)

Documentar no relatório: amostragem, janelamento, FFT, filtros, energia, detecção de transientes, fusão multimodal (futuro).
