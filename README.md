# Lab4 - Rede Neural Simples em Java (Classificação de Dígitos 0 e 1)

Este projeto foi desenvolvido no contexto da unidade curricular de Inteligência Artificial e tem como objetivo a resolução do **Problema 4 - "0 ou 1?"**, baseado na tarefa de reconhecimento de dígitos a partir de imagens simplificadas da base MNIST. O foco está na classificação binária: distinguir se uma imagem representa o dígito 0 ou 1.

A rede neural foi implementada do zero em Java, utilizando perceptrons com função de ativação sigmoide, treinamento com retropropagação e salvamento de pesos e erros para posterior análise.

---

## 🧠 Objetivo do Problema

- Construir e treinar uma rede neural capaz de classificar imagens como representando **o dígito 0 ou 1**.
- As imagens são representadas como vetores com 400 valores (20x20 pixels), normalizados no intervalo [0,1].
- O modelo deve ser capaz de generalizar e ser avaliado num conjunto de teste.

---

## 🛠️ Requisitos
- Java 8 ou superior

---

## 🚀 Como Executar

### 1. Compilar o projeto
Navegue até a pasta `src/` e compile todos os arquivos Java:
```bash
javac *.java
```

### 2. Executar o programa
```bash
java Main
```

---

## 📥 Entrada Esperada

O programa utiliza dois ficheiros CSV como entrada:

### `dataset.csv`
- Contém 800 linhas, cada uma representando uma imagem de 20x20 pixels (total 400 valores por linha).
- Os valores devem estar normalizados no intervalo [0,1].

**Exemplo (linhas simplificadas):**
```
0,0,0.1,...,0.89
0.5,0.6,0.2,...,0.12
```

### `labels.csv`
- Contém os rótulos correspondentes às imagens em `dataset.csv`, um por linha:
  - `0` para o dígito 0
  - `1` para o dígito 1

**Exemplo:**
```
0
1
```

> Os dois ficheiros devem ter o mesmo número de linhas.

---

## 📊 Saída Gerada

- `hidden_weights.csv`, `output_weights.csv`: pesos finais da rede neural após o treino
- `mse.csv`: evolução do erro quadrático médio (MSE) durante as épocas

> Estes ficheiros permitem analisar o desempenho e reusar a rede já treinada.

---

## 📂 Estrutura dos Arquivos

- `Main.java`: ponto de entrada e fluxo de execução principal
- `NeuralNetwork.java`: lógica da rede, inicialização, feedforward e backpropagation
- `Perceptron.java`: representa um neurônio individual com pesos e ativação
- `dataset.csv` / `labels.csv`: dados de treino
- `hidden_weights.csv` / `output_weights.csv`: pesos finais
- `mse.csv`: evolução do erro MSE ao longo do treino

---

## 📈 Estratégias e Técnicas Utilizadas

- Função de ativação: Sigmoide
- Algoritmo de aprendizagem: retropropagação (backpropagation)
- Normalização de dados: [0, 1]
- Divisão dos dados: pode ser feita manualmente para treino/teste (ex: 80% / 20%)
- Detecção de overfitting possível via MSE

---



## 📜 Licença
Este projeto é de uso educacional e livre para modificação e extensão.

---

> Desenvolvido como parte do **TP3** da unidade curricular de Inteligência Artificial, 2024-25 — Problema 4: 0 ou 1?

