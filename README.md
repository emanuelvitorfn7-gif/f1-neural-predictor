<<<<<<< HEAD
# 🏎️ F1 Neural Predictor

Painel interativo em Python que estima a probabilidade de vitória dos pilotos de
Fórmula 1 usando a API [OpenF1](https://openf1.org/), limpeza com Pandas e uma
rede neural inicial.

## Recursos

- Coleta de resultados da OpenF1 desde 2023.
- Limpeza de duplicatas e valores ausentes com Pandas.
- Padronização de nomes de pilotos e equipes.
- Indicadores das últimas cinco corridas: posição, vitórias, pódios, top 5,
  abandono e média de pontos.
- Rede neural `MLPClassifier` com camadas 16 e 8.
- Separação temporal: 80% para treino e 20% para teste.
- Painel com probabilidades, métricas e download em CSV.

## Executar no PowerShell do VS Code

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

O navegador normalmente abrirá em `http://localhost:8501`.

## Tecnologias

- OpenF1
- Pandas e NumPy
- Scikit-learn
- Streamlit
- Plotly

## Estrutura

```text
f1-predictor/
├── app.py
├── predictor.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Método

Cada linha representa um piloto antes de uma corrida. As variáveis são criadas
com `shift(1)`, garantindo que o resultado da corrida prevista não seja usado no
treino daquela própria linha. Como vitórias são raras, os vencedores do conjunto
de treino são reforçados para reduzir o desbalanceamento.

A saída bruta da rede é normalizada entre os pilotos ativos para somar 100%, pois
uma corrida possui um único vencedor. É uma análise educacional, não uma certeza
ou recomendação de apostas.

## Limitações

- A OpenF1 disponibiliza gratuitamente dados históricos principalmente a partir de 2023.
- O modelo não conhece clima, circuito, classificação, atualizações do carro ou acidentes.
- A pontuação usada como variável segue a tabela-base e não inclui bônus excepcionais.
- Uma acurácia alta isoladamente pode enganar, pois há somente um vencedor por corrida.

## Melhorias futuras

- Incluir resultado da classificação e características do circuito.
- Salvar o modelo treinado para reutilização.
- Comparar previsões antigas com os resultados reais.
- Adicionar testes automatizados com respostas simuladas da API.

## Publicar no GitHub

```powershell
git init
git add .
git commit -m "Projeto inicial F1 Neural Predictor"
git branch -M main
git remote add origin https://github.com/SEU-USUARIO/f1-neural-predictor.git
git push -u origin main
```
=======
# f1-neural-predictor
>>>>>>> 3cdd4bb5f6a84a9190c53a6766e8415923b43d3c
