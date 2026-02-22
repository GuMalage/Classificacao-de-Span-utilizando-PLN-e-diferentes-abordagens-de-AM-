##  Classificação de Texto com BERT para a geração de embeddings + Machine Learning

O repositório inclui um experimento completo de classificação de mensagens de texto (spam vs. não spam) utilizando uma abordagem híbrida que combina **Modelos de Linguagem Pré-Treinados (BERT)** com algoritmos clássicos de Machine Learning.

### 🔍 Pipeline Utilizado

O fluxo experimental segue as seguintes etapas:

1. **Carregamento do Dataset**
   - Base de dados de classificação de mensagens (spam/ham).
   - Leitura e organização utilizando Pandas.

2. **Pré-processamento Textual**
   - Limpeza de caracteres especiais
   - Normalização (lowercase)
   - Tokenização com NLTK
   - Remoção de stopwords

3. **Extração de Representações com BERT**
   - Utilização do modelo `bert-base-uncased`
   - Tokenização via `BertTokenizer`
   - Extração do embedding do token `[CLS]`
   - Conversão para vetores numéricos

4. **Normalização dos Vetores**
   - Padronização com `StandardScaler`

5. **Treinamento de Modelos Supervisionados**
   Os embeddings extraídos pelo BERT são utilizados como entrada para diferentes classificadores:

   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Random Forest

6. **Avaliação**
   - Divisão treino/teste com estratificação
   - Cálculo de acurácia

---

## 🤖 Abordagem Metodológica

A estratégia adotada separa o problema em duas camadas:

- **Camada 1 – Representação Semântica:**  
  O modelo BERT é utilizado como extrator de características, capturando informações contextuais profundas do texto.

- **Camada 2 – Classificação:**  
  Algoritmos clássicos de Machine Learning são aplicados sobre os embeddings gerados, permitindo comparar diferentes estratégias de decisão.

Essa abordagem combina o poder semântico dos Transformers com a interpretabilidade e eficiência computacional de modelos tradicionais.

---

## 📊 Modelos Implementados

O repositório contém implementações completas utilizando:

- SVM (Support Vector Machine)
- KNN (K-Nearest Neighbors)
- Decision Tree
- Random Forest

## 🎯 Objetivo do Experimento

Demonstrar como modelos pré-treinados podem ser integrados a algoritmos clássicos de Machine Learning para tarefas de classificação de texto, evidenciando diferenças de desempenho entre abordagens baseadas em distância, margem e árvores de decisão.
