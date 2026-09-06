from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from time import sleep
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_URL = "https://api.openf1.org/v1"
JOLPICA_URL = "https://api.jolpi.ca/ergast/f1"
FEATURES = [
    "media_posicao_5", "vitorias_5", "podios_5", "top5_5",
    "abandono_5", "media_pontos_5",
]
POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


class OpenF1Error(RuntimeError):
    pass


def _retry_delay(error: HTTPError, attempt: int) -> float:
    """Respeita Retry-After e usa espera progressiva quando ele não existe."""
    retry_after = error.headers.get("Retry-After") if error.headers else None
    try:
        return min(max(float(retry_after), 1.0), 30.0)
    except (TypeError, ValueError):
        return min(2.0 ** (attempt + 1), 30.0)


@lru_cache(maxsize=512)
def _request_json(url: str) -> tuple[dict, ...]:
    """Consulta a API com cache e novas tentativas para evitar HTTP 429."""
    request = Request(url, headers={"User-Agent": "f1-neural-predictor/1.0"})
    max_retries = 4
    for attempt in range(max_retries + 1):
        try:
            with urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                if not isinstance(data, list):
                    raise OpenF1Error("A OpenF1 retornou um formato inesperado.")
                return tuple(data)
        except HTTPError as exc:
            if exc.code == 429 and attempt < max_retries:
                sleep(_retry_delay(exc, attempt))
                continue
            if exc.code == 429:
                raise OpenF1Error(
                    "A OpenF1 limitou temporariamente as consultas (HTTP 429). "
                    "O preditor tentou novamente automaticamente; aguarde alguns "
                    "minutos antes de repetir a análise."
                ) from exc
            raise OpenF1Error(f"OpenF1 respondeu com erro HTTP {exc.code}.") from exc
        except (URLError, TimeoutError) as exc:
            raise OpenF1Error(f"Falha de conexão com a OpenF1: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise OpenF1Error("A OpenF1 retornou uma resposta inválida.") from exc
    raise OpenF1Error("Não foi possível consultar a OpenF1 após várias tentativas.")


def _get(endpoint: str, **params) -> list[dict]:
    # A ordenação garante a mesma chave de cache mesmo se os parâmetros forem
    # enviados em ordens diferentes.
    query = urlencode(sorted(params.items()))
    url = f"{BASE_URL}/{endpoint}?{query}"
    return list(_request_json(url))


def _completed_races(year: int) -> list[dict]:
    now = datetime.now().astimezone()
    races = _get("sessions", year=year, session_name="Race")
    valid = []
    for race in races:
        try:
            end = datetime.fromisoformat(race["date_end"].replace("Z", "+00:00"))
            if end <= now:
                valid.append(race)
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(valid, key=lambda item: item["date_end"])


def _clean_name(value: str) -> str:
    """Padroniza nomes vindos em caixa alta ou com espaços duplicados."""
    return " ".join(str(value).strip().split()).title()


def _collect_openf1_data(start_year: int, end_year: int) -> list[dict]:
    rows = []
    for year in range(start_year, end_year + 1):
        races = _completed_races(year)
        if not races:
            continue
        # Uma consulta por temporada basta para nomes e equipes; isso reduz
        # bastante o tempo e a quantidade de chamadas à API.
        latest_key = races[-1]["session_key"]
        drivers = {
            int(d["driver_number"]): d for d in _get("drivers", session_key=latest_key)
        }
        for race_number, race in enumerate(races, start=1):
            session_key = race["session_key"]
            for result in _get("session_result", session_key=session_key):
                number = result.get("driver_number")
                if number is None:
                    continue
                info = drivers.get(int(number), {})
                position = pd.to_numeric(result.get("position"), errors="coerce")
                rows.append({
                    "ano": year,
                    "corrida": race_number,
                    "data": race.get("date_start", ""),
                    "piloto_id": int(number),
                    "piloto": _clean_name(info.get("full_name") or f"Piloto {number}"),
                    "equipe": _clean_name(info.get("team_name") or "Não informado"),
                    "posicao": position,
                    "abandono_api": bool(result.get("dnf") or result.get("dns") or result.get("dsq")),
                })
    return rows


@lru_cache(maxsize=20)
def _jolpica_season(year: int) -> tuple[dict, ...]:
    """Fonte reserva de resultados quando a OpenF1 está indisponível."""
    url = f"{JOLPICA_URL}/{year}/results.json?limit=2000"
    request = Request(url, headers={"User-Agent": "f1-neural-predictor/1.1"})
    try:
        with urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
        races = payload["MRData"]["RaceTable"]["Races"]
        if not isinstance(races, list):
            raise TypeError
        return tuple(races)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError,
            KeyError, TypeError) as exc:
        raise OpenF1Error(
            "Não foi possível obter os resultados nem pela OpenF1 nem pela "
            "fonte histórica reserva. Aguarde alguns minutos e tente novamente."
        ) from exc


def _collect_jolpica_data(start_year: int, end_year: int) -> list[dict]:
    rows = []
    for year in range(start_year, end_year + 1):
        for race in _jolpica_season(year):
            try:
                race_number = int(race["round"])
            except (KeyError, TypeError, ValueError):
                continue
            for result in race.get("Results", []):
                driver = result.get("Driver", {})
                constructor = result.get("Constructor", {})
                status = str(result.get("status", ""))
                try:
                    number = int(result.get("number"))
                except (TypeError, ValueError):
                    continue
                rows.append({
                    "ano": year,
                    "corrida": race_number,
                    "data": race.get("date", ""),
                    "piloto_id": number,
                    "piloto": _clean_name(
                        f'{driver.get("givenName", "")} {driver.get("familyName", "")}'
                    ),
                    "equipe": _clean_name(constructor.get("name") or "Não informado"),
                    "posicao": pd.to_numeric(result.get("position"), errors="coerce"),
                    "abandono_api": status.lower() != "finished" and not status.startswith("+")
                })
    return rows


def collect_data(start_year: int, end_year: int, recent_races: int = 5) -> pd.DataFrame:
    try:
        rows = _collect_openf1_data(start_year, end_year)
    except OpenF1Error:
        # A OpenF1 pode responder 404, 429 ou exigir acesso temporariamente.
        # A fonte reserva fornece os mesmos resultados finais necessários ao modelo.
        rows = _collect_jolpica_data(start_year, end_year)
    if not rows:
        rows = _collect_jolpica_data(start_year, end_year)
    if not rows:
        raise OpenF1Error("Nenhum resultado foi encontrado para o período escolhido.")
    return clean_and_engineer(pd.DataFrame(rows), recent_races)


def clean_and_engineer(data: pd.DataFrame, recent_races: int = 5) -> pd.DataFrame:
    """Limpa duplicatas e cria métricas usando somente corridas anteriores."""
    df = data.copy()
    df = df.drop_duplicates(["ano", "corrida", "piloto_id"], keep="last")
    df["posicao"] = pd.to_numeric(df["posicao"], errors="coerce")
    df["abandono"] = (df["posicao"].isna() | df["abandono_api"].fillna(False)).astype(int)
    df["posicao"] = df["posicao"].fillna(20).clip(1, 20)
    df["venceu"] = (df["posicao"] == 1).astype(int)
    df["podio"] = (df["posicao"] <= 3).astype(int)
    df["top5"] = (df["posicao"] <= 5).astype(int)
    df["pontos"] = df["posicao"].map(POINTS).fillna(0).astype(float)
    df = df.sort_values(["data", "piloto_id"]).reset_index(drop=True)

    window = max(int(recent_races), 1)
    minimum = min(3, window)
    grouped = df.groupby("piloto", sort=False)
    for source, target in [
        ("posicao", "media_posicao_5"), ("venceu", "vitorias_5"),
        ("podio", "podios_5"), ("top5", "top5_5"),
        ("abandono", "abandono_5"), ("pontos", "media_pontos_5"),
    ]:
        # shift(1) impede que o resultado previsto entre nas próprias variáveis.
        df[target] = grouped[source].transform(
            lambda values: values.shift(1).rolling(window, min_periods=minimum).mean()
        )
    return df.dropna(subset=FEATURES).reset_index(drop=True)


def train_model(df: pd.DataFrame):
    if len(df) < 40 or df["venceu"].nunique() < 2:
        raise OpenF1Error("Ainda não existem corridas suficientes para treinar o modelo.")

    split = max(int(len(df) * 0.8), 1)
    train, test = df.iloc[:split], df.iloc[split:]
    winners = train[train["venceu"] == 1]
    # Reforça os raros casos de vitória para reduzir o desbalanceamento.
    balanced = pd.concat([train, *([winners] * 8)], ignore_index=True)

    model = Pipeline([
        ("escala", StandardScaler()),
        ("rede_neural", MLPClassifier(
            hidden_layer_sizes=(16, 8), activation="relu", solver="adam",
            alpha=0.01, max_iter=800, random_state=42, early_stopping=True,
        )),
    ])
    model.fit(balanced[FEATURES], balanced["venceu"])

    prediction = model.predict(test[FEATURES])
    probability = model.predict_proba(test[FEATURES])[:, 1]
    metrics = {
        "acuracia": accuracy_score(test["venceu"], prediction),
        "precisao": precision_score(test["venceu"], prediction, zero_division=0),
        "recall": recall_score(test["venceu"], prediction, zero_division=0),
        "f1": f1_score(test["venceu"], prediction, zero_division=0),
        "auc": roc_auc_score(test["venceu"], probability) if test["venceu"].nunique() == 2 else None,
        "matriz": confusion_matrix(test["venceu"], prediction, labels=[0, 1]).tolist(),
        "treino": len(train),
        "teste": len(test),
    }
    return model, metrics


def predict_next_race(df: pd.DataFrame, model: Pipeline, recent_races: int) -> pd.DataFrame:
    latest = (
        df.sort_values("data").groupby("piloto", as_index=False).tail(1).copy()
    )
    latest_year = df["ano"].max()
    season = df[df["ano"] == latest_year]
    cutoff = season["corrida"].max() - recent_races + 1
    active_names = season.loc[season["corrida"] >= cutoff, "piloto"].unique()
    latest = latest[latest["piloto"].isin(active_names)]
    if latest.empty:
        raise OpenF1Error("Nenhum piloto ativo foi encontrado nas corridas recentes.")
    raw = model.predict_proba(latest[FEATURES])[:, 1]
    total = float(np.sum(raw))
    if total <= 0 or not np.isfinite(total):
        raise OpenF1Error("O modelo não gerou probabilidades válidas. Tente ampliar o período.")
    # Em cada GP existe um vencedor; normalização facilita a leitura do painel.
    latest["probabilidade"] = raw / total * 100
    return latest[["piloto", "equipe", *FEATURES, "probabilidade"]].sort_values(
        "probabilidade", ascending=False
    ).reset_index(drop=True).round(2)


def run_analysis(start_year: int, end_year: int, recent_races: int):
    dataset = collect_data(start_year, end_year, recent_races)
    model, metrics = train_model(dataset)
    probabilities = predict_next_race(dataset, model, recent_races)
    return probabilities, metrics, dataset
