# 🛰️🔥 OrbitalFire — Detecção de Queimadas em Imagens de Satélite (ACV)

**Disciplina:** Applied Computer Vision (ACV)
**Global Solution 2026 · 1º Semestre · Indústria Espacial**
**FIAP · Engenharia de Software · 4º Ano · ODS 13 (Ação Climática)**

> Integrantes:
> - Bruno Eduardo Caputo Paulino — RM 558303

Módulo de **Visão Computacional** do OrbitalFire: classifica imagens de satélite em
`wildfire` (com indício de queimada) ou `nowildfire`, usando **duas CNNs treinadas do
zero** (sem modelos pré-treinados). Atua como validação visual dos alertas gerados
pelo modelo de risco (GAIE) e pelo pipeline de dados (BDDI).

---

## 1. Problema

Classificação binária de imagens de satélite para confirmar/descartar indício de
queimada, reduzindo falsos positivos antes do acionamento operacional. Entrada =
imagem de observação da Terra (conexão direta com a Indústria Espacial).

## 2. Dataset

**Wildfire Prediction Dataset (Satellite Images)** — Kaggle
(`abdelghaniaaba/wildfire-prediction-dataset`): ~42 mil imagens RGB (~350×350),
2 classes (`wildfire` / `nowildfire`), já dividido em `train / valid / test`.

Pré-processamento: resize 128×128, normalização `[0,1]`, data augmentation (flip,
rotação, zoom) aplicado igualmente aos dois modelos.

## 3. Arquiteturas (treinadas do zero)

| Modelo | Estrutura | Característica |
|---|---|---|
| **CNN-Base** | 3 blocos Conv→MaxPool + Flatten + Dense | Baseline simples (~4,3M params) |
| **CNN-Plus** | 4 blocos com BatchNorm + GlobalAveragePooling + Dropout | Mais profunda e regularizada (~1,2M params) |

A mesma augmentation é aplicada aos dois — a diferença é **puramente arquitetural**,
o que permite justificar tecnicamente o impacto das mudanças.

## 4. Resultados

> Preencher após executar o notebook (os números saem das células de avaliação).
> Meta de referência: **≥ 88%** de acurácia no teste.

| Modelo | Acurácia (teste) | Loss | Atingiu 88% |
|---|---|---|---|
| CNN-Base | _preencher_ | _preencher_ | _preencher_ |
| CNN-Plus | _preencher_ | _preencher_ | _preencher_ |

Inclui no notebook: curvas de acurácia/loss por época, matriz de confusão,
`classification_report`, exemplos de acertos/erros e tabela comparativa.

## 5. Estrutura do repositório

```
orbitalfire-acv/
├── notebooks/orbitalfire_acv.ipynb   # pipeline completo (treino, avaliação, demo)
├── src/architectures.py              # definição das 2 CNNs (arquivo de arquitetura)
├── app/app_gradio.py                 # demonstração funcional
├── sample_images/                    # algumas imagens de amostra (adicionar)
├── requirements.txt
├── .gitignore
└── README.md
```

## 6. Como executar

### Opção A — Kaggle (recomendado, tem GPU grátis)
1. Crie um Notebook no Kaggle e faça **Add Data → Wildfire Prediction Dataset**.
2. Ative a GPU (Settings → Accelerator → GPU).
3. Faça upload de `notebooks/orbitalfire_acv.ipynb` e execute todas as células.

### Opção B — Google Colab
1. Ative a GPU (Ambiente de execução → Alterar tipo).
2. Descomente o bloco `kagglehub` na célula de localização do dataset
   (precisa do seu token Kaggle).
3. Execute o notebook.

### Demonstração (após treinar e salvar `best_model.keras`)
```bash
pip install -r requirements.txt
python app/app_gradio.py
```

## 7. Entrega

- ✅ Repositório GitHub público com notebook `.ipynb`, `architectures.py`, app e README
- ✅ 2 CNNs do zero, treino/validação/teste, métricas e comparação
- ⬜ **Vídeo de até 3 min** no YouTube demonstrando a solução → colar o link aqui
- ⬜ Preencher resultados (seção 4) e completar nomes/RM dos integrantes
