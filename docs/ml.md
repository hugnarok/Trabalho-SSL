# Pipeline de Machine Learning — SafeAlert

Este documento detalha o pipeline de classificação acústica baseado em aprendizado de máquina do SafeAlert: desde a extração de features até a inferência em tempo real no dispositivo de borda.

---

## Visão Geral do Pipeline

```
Áudio bruto (PCM, 16 kHz, mono)
        │
        ▼
  Janelamento de Hann (N_FFT=512, Hop=256)
        │
        ▼
  STFT → espectro de potência
        │
        ▼
  Banco de 64 filtros mel (20 Hz – 8.000 Hz)
        │
        ▼
  Log-compressão + normalização z-score
        │
        ▼
  Tensor 1 × 64 × 128  (canais × filtros × quadros)
        │
        ▼
  CNN leve (3 blocos Conv → Dense 128 → Softmax 3)
        │
        ▼
  [normal | grito | impacto]  +  probabilidade
```

---

## Extração de Features (`client/ml/features.py`)

### Por que mel-espectrograma?

O ouvido humano não percebe frequências de forma linear: diferenças em frequências baixas são muito mais perceptíveis do que as mesmas diferenças em frequências altas. A escala mel aproxima essa curva de percepção. Ao projetar o espectro de potência nessa escala, o modelo CNN recebe uma representação que enfatiza as regiões espectrais mais relevantes para a percepção de eventos sonoros.

### Parâmetros utilizados

| Parâmetro | Valor | Justificativa |
|---|---|---|
| Taxa de amostragem | 16.000 Hz | Cobre frequências até 8 kHz — região da voz e gritos humanos |
| Tamanho da janela (`N_FFT`) | 512 amostras (32 ms) | Resolução espectral adequada para sons ambientais |
| Salto (`Hop`) | 256 amostras (16 ms) | 50% de sobreposição — equilíbrio entre resolução temporal e custo |
| Janela | Hann | Atenua vazamento espectral nas bordas de cada quadro |
| Número de filtros mel | 64 | Resolução mel suficiente para distinguir grito/impacto/normal |
| Duração do segmento | 5 s | Gera tensor 64 × 128 quadros (fixo, independente do hardware) |

### Normalização

O espectrograma log-mel resultante é normalizado por z-score (média 0, desvio padrão 1) calculado sobre o próprio segmento. Isso torna o modelo robusto a variações de volume entre ambientes com níveis de ruído muito diferentes.

### Implementação

O banco de filtros mel é calculado analiticamente pela função `_get_mel_fb()` e cacheado como variável global após a primeira chamada. Isso elimina o recálculo a cada janela de áudio e remove a dependência de `librosa` em tempo de inferência, reduzindo o tamanho da instalação no dispositivo de borda.

---

## Arquitetura CNN (`client/ml/model_arch.py`)

A rede foi projetada para inferência em borda: pequena o suficiente para rodar em CPU sem acelerador dedicado, com latência abaixo de 50 ms.

```
Entrada: 1 × 64 × 128

Bloco 1: Conv2d(1→16, 3×3) → BatchNorm → ReLU → MaxPool2d(2×2)
         Saída: 16 × 32 × 64

Bloco 2: Conv2d(16→32, 3×3) → BatchNorm → ReLU → MaxPool2d(2×2)
         Saída: 32 × 16 × 32

Bloco 3: Conv2d(32→64, 3×3) → BatchNorm → ReLU → AdaptiveAvgPool2d(4×4)
         Saída: 64 × 4 × 4 = 1.024 valores

Flatten → Linear(1024→128) → Dropout(0.3) → ReLU

Saída: Linear(128→3) → Softmax
       [P(normal), P(grito), P(impacto)]
```

**Decisões de arquitetura:**

- **BatchNorm** após cada convolução: estabiliza o treinamento com poucos dados e reduz a necessidade de ajuste fino da taxa de aprendizado.
- **AdaptiveAvgPool2d(4,4)** no terceiro bloco (em vez de MaxPool fixo): garante que o tensor de saída tenha dimensão fixa independente do tamanho de entrada — importante para robustez a segmentos de duração levemente diferente.
- **Dropout(0.3)** na camada densa: regularização para dataset pequeno.
- Tamanho total: ~500 KB em disco (float32). Cabe em memória de qualquer dispositivo com mais de 50 MB de RAM disponível.

---

## Dataset NIGENS

**NIGENS** (NIGENS General Sound Events Database) é um corpus público de eventos sonoros gerais, distribuído sob licença **CC BY-NC-ND 4.0** (uso não comercial, sem modificação, com atribuição). Disponível em: https://zenodo.org/records/2535878

### Mapeamento de classes

| Pastas do NIGENS | Classe SafeAlert | Justificativa |
|---|---|---|
| `femaleScream`, `maleScream` | `grito` | Gritos de socorro femininos e masculinos |
| `crash`, `knock` | `impacto` | Sons de impacto, queda, batida |
| `femaleSpeech`, `maleSpeech` | `normal` | Fala cotidiana — evita que voz dispare alerta |
| `general`, `piano`, `footsteps` | `normal` | Sons ambiente comuns em campus |
| `phone`, `engine`, `alarm` | `normal` | Reduz falsos positivos de ruído mecânico |
| `fire`, `dog`, `baby` | `normal` | Sons biologicamente salientes mas inócuos |

### Desbalanceamento de classes

O NIGENS tem muito mais amostras de sons normais do que de gritos e impactos. No conjunto de validação utilizado para avaliar o modelo, havia 110 amostras normais, 14 de grito e 10 de impacto. Esse desbalanceamento infla a acurácia global e prejudica o recall das classes de risco — ver seção de Resultados.

**Mitigação implementada:** a classe `normal` é construída com múltiplas subcategorias para aumentar a diversidade acústica e reduzir o risco de sobreajuste ao estilo específico de ruído do NIGENS.

**Mitigação pendente:** adicionar `weight` ao `CrossEntropyLoss` inversamente proporcional à frequência de cada classe seria a correção mais impactante para o F1 de grito e impacto.

---

## Passo a Passo: Treino

### Opção 1 — Download e treino automático (recomendado)

```bash
# Baixa NIGENS (~2 GB), organiza os clipes e treina a CNN em um comando:
python -m scripts.download_nigens --all --epochs 25
```

Internamente executa três etapas em sequência:
1. `download_nigens.py` — baixa e extrai o ZIP do Zenodo
2. `prepare_nigens.py` — copia e renomeia os clipes para `data/datasets/prepared/grito/`, `impacto/`, `normal/`
3. `train_sound_model.py` — treina a CNN e salva os pesos

### Opção 2 — Etapas separadas

```bash
# 1. Só download e extração:
python -m scripts.download_nigens

# 2. Organização dos clipes (necessário após o download):
python -m scripts.download_nigens --prepare

# 3. Treinamento:
python -m scripts.train_sound_model --epochs 25 --batch-size 16 --lr 0.001
```

### Opção 3 — Download manual (se o automático falhar)

1. Acesse https://zenodo.org/records/2535878 e baixe `NIGENS.zip`
2. Coloque em `data/datasets/nigens/downloads/NIGENS.zip`
3. Execute:
   ```bash
   python -m scripts.download_nigens --skip-download --all --epochs 25
   ```

> **macOS:** a extração de ZIP usa `unzip` do sistema (necessário para suporte a Deflate64). Verifique com `which unzip`.

### Opção 4 — Áudios próprios

Coloque arquivos `.wav` (mono, qualquer taxa de amostragem — será reamostrado para 16 kHz) em:

```
data/samples/ml/grito/
data/samples/ml/impacto/
data/samples/ml/normal/
```

Os dois diretórios (`datasets/prepared/` e `samples/ml/`) são combinados automaticamente pelo script de treino. Isso permite aumentar a representação de classes raras com gravações próprias do ambiente-alvo.

### Parâmetros do script de treino

```bash
python -m scripts.train_sound_model \
  --epochs 25 \        # número de épocas
  --batch-size 16 \    # tamanho do batch
  --lr 0.001 \         # taxa de aprendizado (Adam)
  --val-split 0.15 \   # fração de validação
  --test-split 0.15    # fração de teste
```

O script imprime métricas por época (loss de treino e validação) e, ao final, o relatório completo de classificação (precisão, recall, F1 por classe) sobre o conjunto de teste.

### Artefatos gerados

| Arquivo | Descrição |
|---|---|
| `data/models/sound_classifier.pt` | Pesos da CNN (state dict PyTorch) |
| `data/models/sound_classifier.json` | Metadados: classes, taxa de amostragem, acurácia final |

---

## Avaliação do Modelo

```bash
python -m scripts.evaluate_model
```

Exibe:
- Acurácia global no conjunto de teste
- Relatório por classe (precisão, recall, F1, suporte)
- Matriz de confusão em texto

Use para verificar se o modelo está generalizando antes de colocá-lo em produção.

---

## Inferência em Tempo Real (`client/ml/classifier.py`)

O classificador é carregado como **singleton** com _lazy loading_: o arquivo de pesos só é lido do disco na primeira chamada, não na inicialização do cliente. Isso permite que o cliente inicie mesmo se o modelo ainda não foi treinado — a heurística de energia assume o papel de fallback.

Fluxo de inferência por ciclo:

1. `DetectionWorker` chama `detect_ml_events(audio, sample_rate)`
2. `ml_events.py` chama `SoundClassifier.predict(audio, sr)`
3. `classifier.py` extrai o mel-espectrograma via `features.log_mel_spectrogram()`
4. Passa o tensor normalizado pela CNN
5. Obtém probabilidades softmax
6. Se `max(probs) >= ML_CONFIDENCE_THRESHOLD` e classe vencedora ≠ `normal`, retorna `DetectionResult` com tipo e confiança
7. Caso contrário, retorna `DetectionResult(detected=False)`

Latência típica (CPU, sem GPU): < 50 ms em x86-64 (Intel Core i7) e ARM M1.

---

## Resultados Obtidos

| Classe | Precisão | Recall | F1-Score | N amostras |
|---|---|---|---|---|
| Normal | 0,88 | 0,95 | 0,91 | 110 |
| Grito | 0,82 | 0,64 | 0,72 | 14 |
| Impacto | 0,20 | 0,10 | 0,13 | 10 |
| **Média ponderada** | **0,82** | **0,85** | **0,83** | **134** |

**Interpretação:**

- A acurácia global de 85,07% é enganosa — reflete principalmente o desempenho na classe majoritária (normal).
- O recall de 0,64 para grito significa que 36% dos gritos reais não são detectados pela CNN. Para um sistema de segurança, isso é inaceitável como detector único.
- O F1 de 0,13 para impacto indica falha quase completa na generalização para essa classe, dada a escassez de amostras de treino.
- Esses resultados justificam a manutenção da heurística de energia como fallback obrigatório.

**Nota sobre variância:** com apenas 10 amostras de impacto no conjunto de teste, uma única amostra a mais ou a menos altera o F1 em ~0,10. Os valores devem ser interpretados como indicativos de tendência.

---

## Próximas Melhorias

### Alta prioridade (impacto direto nos resultados)

1. **Pesos de classe no loss:** `CrossEntropyLoss(weight=torch.tensor([1/110, 1/14, 1/10]))` — penaliza mais os erros nas classes raras sem precisar de mais dados.

2. **Early stopping:** salvar os pesos quando `val_loss` é mínimo, não os do último epoch. Evita overfitting em datasets pequenos.

3. **Data augmentation:** aplicar sobre as classes minoritárias:
   - _Time stretching_ (±10%): altera velocidade sem mudar tom
   - _Pitch shifting_ (±2 semitons): varia frequência fundamental
   - Adição de ruído gaussiano (SNR 10–20 dB): simula ambientes ruidosos
   - _SpecAugment_: mascara faixas de tempo e frequência no espectrograma

4. **Coleta de dados reais:** gravar gritos e impactos no ambiente-alvo real (corredor, estacionamento do campus) para reduzir a distância de domínio entre o NIGENS e o contexto de implantação.

### Média prioridade

5. **Separação de stride vs. contador:** manter contadores de streak independentes por tipo de evento em vez de um único contador global — evita que alternância rápida entre detectores impeça qualquer confirmação.

6. **Fusão com prioridade semântica:** `HELP_REQUEST > SCREAM > IMPACT` como critério de desempate além da confiança, refletindo a especificidade semântica de cada evento.

### Baixa prioridade

7. **Integração de visão computacional:** estimação de pose humana (MoveNet, OpenPose) para detectar posturas de agressão — fusão multimodal verdadeira de áudio e vídeo.
