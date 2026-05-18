# ==============================================================================
# 🤖 CAMPO MINADO - IA SWEEPER
# Objetivo: Treinar uma Rede Neural para jogar Campo Minado
# ==============================================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

# configurações dos caminhos, e após o treino compeltar salva os modelos treinados nesse caminho
DATA_PATH = Path("data/minesweeper_data.csv")
FALLBACK_DATA_PATH = Path("dataset/minesweeper_dataset/minesweeper_dataset.csv")
MODEL_PATH = Path("minesweeper_modelo.keras")
SCALER_PATH = Path("scaler.pkl")

# evita ter que carregar o scaler e o modelo toda vez que a previsão for feita
_modelo_cache = None
_scaler_cache = None

# define a função pra carregar o data set do campo e se nao achar ele avisa pro usuario colocar o data set no caminho correto
def _carregar_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    if FALLBACK_DATA_PATH.exists():
        return pd.read_csv(FALLBACK_DATA_PATH)
    raise FileNotFoundError(
        "erro ao carregar o data set"
    )

# constroi o modelo de rede com 3 camadas ocultas de 256 neuronios e com saida binaria pois é segura ou não segura. 
def _construir_modelo(input_dim):
    modelo = Sequential(
        [
            Dense(256, activation="relu", input_shape=(input_dim,)),
            Dense(256, activation="relu"),
            Dense(256, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    modelo.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return modelo

# aqui é a função principal do treino, carrega os dados e treina o modelo e depois salva o modelo e o scaler
def treinar_e_salvar_modelo():
    print("[TREINO] Carregando dados")
    df = _carregar_dataset()

    X = df.drop(columns=["safe"]).values
    y = df["safe"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
# normaliza os dados usando o standard scaler, pra melhorar a performance do modelo pra prever as jogasdas seguras
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
# começa o treino do modelo e define o numero das epochs e o batch size e depois avalia a acuracia do modelo
    modelo = _construir_modelo(X_train.shape[1])
    print("[TREINO] Treinando modelo")
    modelo.fit(
        X_train,
        y_train,
        epochs=15,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
    )
# avalia o modelo usando os dados de teste e mostra a acuracia e a loss do modelo 
    loss, acuracia = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"[TREINO] Acuracia teste: {acuracia * 100:.2f}% | Loss: {loss:.4f}")
# salva o modelo no caminho que foi definido no inicio do codigo
    modelo.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[TREINO] Modelo salvo em: {MODEL_PATH}")
    print(f"[TREINO] Scaler salvo em: {SCALER_PATH}")

    return modelo, scaler

# se nao tem nenhum modelo salvo ele avisa o usuario pra treinar o modelo e depois salva  no cache
def _carregar_modelo():
    global _modelo_cache
    if _modelo_cache is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Modelo nao encontrado, treine e salve o modelo")
        _modelo_cache = load_model(MODEL_PATH)
    return _modelo_cache

# faz a mesma coisa do modelo, se nao tiver o scaler ele pede pra treinar antes e depois salva no cahce dele
def _carregar_scaler():
    global _scaler_cache
    if _scaler_cache is None:
        if not SCALER_PATH.exists():
            raise FileNotFoundError("Scaler nao encontrado treine e salve o modelo.")
        _scaler_cache = joblib.load(SCALER_PATH)
    return _scaler_cache

#  calcula a densidade, que é  a proporcao de celulas desconhecidas em relacao ao tabuleiro inteiro
def _densidade_global(tabuleiro):
    total = tabuleiro.size
    if total == 0:
        return 0.0
    desconhecidos = np.sum(tabuleiro == -1)
    return float(desconhecidos) / float(total)

# ela pega uma celula e verifica as celulas ao redor dela pra verfificar se não tem  minas
def _extrair_features(tabuleiro, linha, coluna):
    linhas, colunas = tabuleiro.shape
    features = []

    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if dx == 0 and dy == 0:
                continue
            nx = linha + dx
            ny = coluna + dy
            if 0 <= nx < linhas and 0 <= ny < colunas:
                features.append(tabuleiro[nx, ny])
            else:
                features.append(-1)

    features.append(_densidade_global(tabuleiro))
    return np.array(features, dtype=float)


def _primeira_desconhecida(tabuleiro):
    posicoes = np.argwhere(tabuleiro == -1)
    if posicoes.size == 0:
        return (0, 0)
    linha, coluna = posicoes[0]
    return (int(linha), int(coluna))


# ============ FUNCAO OBRIGATORIA ============
def prever_jogada_segura(tabuleiro_atual, modelo):
    """
    Recebe o estado atual do tabuleiro e retorna a melhor jogada.

    Parametros:
        tabuleiro_atual: matriz 2D do tabuleiro (valores -1 para desconhecido)
        modelo: modelo treinado (opcional)

    Retorno:
        (x, y): tupla com coordenadas (linha, coluna)
    """
    tabuleiro = np.array(tabuleiro_atual)

    if tabuleiro.ndim != 2:
        return (0, 0)

    try:
        modelo_final = modelo or _carregar_modelo()
        scaler = _carregar_scaler()
    except FileNotFoundError:
        return _primeira_desconhecida(tabuleiro)

    candidatos = np.argwhere(tabuleiro == -1)
    if candidatos.size == 0:
        return (0, 0)
# aqui ele percorre as celulas desconhecidas e extrai as features pra cada uma delas e depois usa o modelo pra prever se é segura ou não e escolhe a que tem a maior probabilidade de ser segura
    melhor_jogada = None
    melhor_score = -1.0

    for linha, coluna in candidatos:
        features = _extrair_features(tabuleiro, int(linha), int(coluna))
        features_norm = scaler.transform([features])
        score = float(modelo_final.predict(features_norm, verbose=0)[0][0])
        if score > melhor_score:
            melhor_score = score
            melhor_jogada = (int(linha), int(coluna))
# ela retorna a melhor jogada se não tiver nenhuma jogada segura ele retorna a primeira celula desconhecida
    return melhor_jogada or (0, 0)


if __name__ == "__main__":
    treinar_e_salvar_modelo()
