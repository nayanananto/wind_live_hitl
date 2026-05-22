from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path, PureWindowsPath
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.preprocessing.feature_engineering import prepare_semantic_frame
from app.semantic.encoders.statistical_encoder import StatisticalWindowEncoder
from app.semantic.summarizers.physics_summary import summarize_window
from app.semantic.tokenization.cluster_tokenizer import ClusterTokenizer
from app.semantic.windowing.window_builder import SemanticWindow


META_COLUMNS = {
    "window_id",
    "window_start",
    "window_end",
    "window_rows",
    "time_span_hours",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(raw_path: str | None, fallback_dirs: list[Path]) -> Path:
    if not raw_path:
        raise ValueError("Missing artifact path in semantic metadata.")
    path = Path(raw_path)
    if path.exists():
        return path

    basename = path.name
    if "\\" in raw_path:
        basename = PureWindowsPath(raw_path).name
    for directory in fallback_dirs:
        candidate = PROJECT_ROOT / directory / basename
        if candidate.exists():
            return candidate

    matches = list(PROJECT_ROOT.glob(f"**/{basename}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve artifact path: {raw_path}")


def _read_live_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"No live observations found in {path}")

    rename_map = {
        "observation_time_utc": "datetime",
        "wind_gust": "wind_gust_10m_ms",
    }
    frame = frame.rename(columns={old: new for old, new in rename_map.items() if old in frame.columns})
    if "datetime" not in frame.columns:
        raise ValueError("Live CSV must contain observation_time_utc or datetime.")
    if "wind_speed" not in frame.columns:
        raise ValueError("Live CSV must contain wind_speed in m/s.")

    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates(
        subset=["datetime", "station_id"],
        keep="last",
    )
    if frame.empty:
        raise ValueError("No valid live rows remained after datetime parsing.")
    return frame.reset_index(drop=True)


def _feature_row_from_window(
    frame: pd.DataFrame,
    station: str,
    row_start: int,
    row_end: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    window_frame = frame.iloc[row_start : row_end + 1].copy()
    prepared = prepare_semantic_frame(window_frame)
    start = pd.Timestamp(prepared["datetime"].iloc[0])
    end = pd.Timestamp(prepared["datetime"].iloc[-1])
    window_id = f"live_{station}_{start.strftime('%Y%m%d%H%M')}_{end.strftime('%Y%m%d%H%M')}"
    window = SemanticWindow(
        window_id=window_id,
        start_time=start,
        end_time=end,
        row_start=row_start,
        row_end=row_end,
        frame=prepared,
    )
    row = summarize_window(window)
    diagnostics = {
        "live_window_rows_used": int(len(prepared)),
        "live_window_start": start.isoformat(),
        "live_window_end": end.isoformat(),
        "live_window_span_hours": float((end - start).total_seconds() / 3600.0),
    }
    return row, diagnostics


def _build_feature_rows(
    frame: pd.DataFrame,
    station: str,
    live_window_rows: int,
    live_step_rows: int,
    backfill: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if frame.empty:
        raise ValueError("Cannot build live semantic windows from an empty frame.")

    window_size = max(1, int(live_window_rows))
    step = max(1, int(live_step_rows))
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    if len(frame) < window_size:
        row, diag = _feature_row_from_window(frame, station=station, row_start=0, row_end=len(frame) - 1)
        rows.append(row)
        diagnostics.append(diag)
    elif backfill:
        for end_idx in range(window_size - 1, len(frame), step):
            start_idx = end_idx - window_size + 1
            row, diag = _feature_row_from_window(frame, station=station, row_start=start_idx, row_end=end_idx)
            rows.append(row)
            diagnostics.append(diag)
    else:
        row, diag = _feature_row_from_window(
            frame,
            station=station,
            row_start=len(frame) - window_size,
            row_end=len(frame) - 1,
        )
        rows.append(row)
        diagnostics.append(diag)

    for diag in diagnostics:
        diag["live_window_rows_requested"] = int(window_size)
        diag["live_step_rows"] = int(step)
    return pd.DataFrame(rows), diagnostics


def _load_cluster_profiles(metadata: dict[str, Any]) -> dict[int, dict[str, Any]]:
    path = _resolve_path(
        metadata["output_paths"].get("cluster_profiles"),
        [Path("data/semantic")],
    )
    frame = pd.read_csv(path)
    if frame.empty or "token_id" not in frame.columns:
        return {}
    return {
        int(row["token_id"]): row.to_dict()
        for _, row in frame.iterrows()
        if pd.notna(row.get("token_id"))
    }


def _load_states(metadata: dict[str, Any]) -> pd.DataFrame:
    path = _resolve_path(
        metadata["output_paths"].get("semantic_states"),
        [Path("data/semantic")],
    )
    frame = pd.read_csv(path)
    if "window_start" in frame.columns:
        frame["window_start"] = pd.to_datetime(frame["window_start"], errors="coerce")
        frame = frame.sort_values("window_start").reset_index(drop=True)
    return frame


def _similar_windows(live_embedding: np.ndarray, states: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    embedding_columns = [column for column in states.columns if column.startswith("embedding_")]
    if not embedding_columns:
        return []
    matrix = states[embedding_columns].to_numpy(dtype=float)
    query = np.asarray(live_embedding, dtype=float).reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1) * max(float(np.linalg.norm(query)), 1e-12)
    distances = 1.0 - ((matrix @ query.ravel()) / np.maximum(norms, 1e-12))
    order = np.argsort(distances)[: max(1, int(top_k))]
    rows: list[dict[str, Any]] = []
    keep = [
        "window_id",
        "window_start",
        "window_end",
        "token_id",
        "regime_name",
        "wind_speed_mean",
        "wind_speed_std",
        "wind_speed_min",
        "wind_speed_max",
        "gust_factor",
        "direction_abs_change_mean_deg",
    ]
    for rank, idx in enumerate(order, start=1):
        row = states.iloc[int(idx)]
        payload = {key: row.get(key) for key in keep if key in row}
        payload["rank"] = rank
        payload["retrieval_distance"] = float(distances[int(idx)])
        rows.append(payload)
    return rows


def _phase_candidates_from_token(
    token_id: int,
    states: pd.DataFrame,
    profiles: dict[int, dict[str, Any]],
    horizon_steps: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], int]:
    tokens = pd.to_numeric(states["token_id"], errors="coerce").dropna().astype(int).to_numpy()
    horizon = max(1, int(horizon_steps))
    counts: Counter[int] = Counter()
    for idx in range(0, len(tokens) - horizon):
        if int(tokens[idx]) == int(token_id):
            counts[int(tokens[idx + horizon])] += 1
    support = sum(counts.values())
    if not support:
        return [], 0

    rows: list[dict[str, Any]] = []
    for rank, (candidate, count) in enumerate(counts.most_common(max(1, int(top_k))), start=1):
        profile = profiles.get(int(candidate), {})
        rows.append(
            {
                "rank": rank,
                "token_id": int(candidate),
                "regime_name": str(profile.get("regime_name") or f"token {candidate}"),
                "probability": round(float(count / support), 4),
                "count": int(count),
                "support": int(support),
                "transition_source": "live_token_markov",
            }
        )
    return rows, support


def _phase_candidates_from_sequence(
    token_sequence: list[int],
    states: pd.DataFrame,
    profiles: dict[int, dict[str, Any]],
    horizon_steps: int,
    top_k: int,
    min_support: int,
) -> tuple[list[dict[str, Any]], int, str]:
    sequence = [int(token) for token in token_sequence if pd.notna(token)]
    if not sequence:
        return [], 0, "no_live_sequence"

    tokens = pd.to_numeric(states["token_id"], errors="coerce").dropna().astype(int).to_numpy()
    horizon = max(1, int(horizon_steps))
    threshold = max(1, int(min_support))
    fallback: tuple[list[int], Counter[int], int, str] | None = None

    for length in range(len(sequence), 1, -1):
        suffix = sequence[-length:]
        counts: Counter[int] = Counter()
        for end_idx in range(length - 1, len(tokens) - horizon):
            past = [int(x) for x in tokens[end_idx - length + 1 : end_idx + 1]]
            if past == suffix:
                counts[int(tokens[end_idx + horizon])] += 1
        support = sum(counts.values())
        if support:
            source = "live_exact_sequence" if length == len(sequence) else f"live_suffix_{length}_tokens"
            if fallback is None:
                fallback = (suffix, counts, support, f"{source}_low_support")
            if support >= threshold:
                return _rank_phase_counts(counts, support, source, profiles, top_k), support, source

    candidates, support = _phase_candidates_from_token(
        token_id=sequence[-1],
        states=states,
        profiles=profiles,
        horizon_steps=horizon,
        top_k=top_k,
    )
    if support >= threshold:
        return candidates, support, "live_last_token_markov"
    if fallback is not None:
        _, counts, support, source = fallback
        return _rank_phase_counts(counts, support, source, profiles, top_k), support, source
    return candidates, support, "live_last_token_markov_low_support"


def _rank_phase_counts(
    counts: Counter[int],
    support: int,
    source: str,
    profiles: dict[int, dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not support:
        return rows
    for rank, (candidate, count) in enumerate(counts.most_common(max(1, int(top_k))), start=1):
        profile = profiles.get(int(candidate), {})
        rows.append(
            {
                "rank": rank,
                "token_id": int(candidate),
                "regime_name": str(profile.get("regime_name") or f"token {candidate}"),
                "probability": round(float(count / support), 4),
                "count": int(count),
                "support": int(support),
                "transition_source": source,
            }
        )
    return rows


def _append_history(history_path: Path, rows: pd.DataFrame) -> pd.DataFrame:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        history = pd.read_csv(history_path)
        combined = pd.concat([history, rows], ignore_index=True)
    else:
        combined = rows.copy()
    combined = combined.drop_duplicates(subset=["window_id"], keep="last")
    if "window_end" in combined.columns:
        combined["window_end_sort"] = pd.to_datetime(combined["window_end"], errors="coerce", utc=True)
        combined = combined.sort_values("window_end_sort").drop(columns=["window_end_sort"]).reset_index(drop=True)
    combined.to_csv(history_path, index=False)
    return combined


def build_live_semantic_state(
    live_csv: Path,
    metadata_path: Path,
    station: str,
    live_window_rows: int,
    live_step_rows: int,
    history_length: int,
    min_support: int,
    backfill: bool,
    top_k: int,
    horizon_steps: int,
    output_dir: Path,
) -> dict[str, Any]:
    metadata = _load_json(metadata_path)
    feature_columns = [column for column in metadata.get("feature_columns", []) if column not in META_COLUMNS]
    if not feature_columns:
        raise ValueError("Semantic metadata does not contain feature_columns.")

    encoder_path = _resolve_path(metadata["output_paths"].get("encoder"), [Path("artifacts/semantic_encoder")])
    tokenizer_path = _resolve_path(metadata["output_paths"].get("tokenizer"), [Path("artifacts/tokenizer")])
    encoder = StatisticalWindowEncoder.load(str(encoder_path))
    tokenizer = ClusterTokenizer.load(str(tokenizer_path))

    live_rows = _read_live_rows(live_csv)
    feature_rows, diagnostics_rows = _build_feature_rows(
        live_rows,
        station=station,
        live_window_rows=live_window_rows,
        live_step_rows=live_step_rows,
        backfill=backfill,
    )
    matrix = feature_rows.drop(columns=list(META_COLUMNS), errors="ignore").reindex(columns=feature_columns)
    embedding = encoder.transform(matrix)
    token_frame = tokenizer.transform(embedding)

    profiles = _load_cluster_profiles(metadata)
    states = _load_states(metadata)
    embedding_columns = [f"embedding_{idx:02d}" for idx in range(embedding.shape[1])]
    live_states = pd.concat(
        [
            feature_rows.reset_index(drop=True),
            pd.DataFrame(embedding, columns=embedding_columns),
            token_frame.reset_index(drop=True),
        ],
        axis=1,
    )
    live_states["token_id"] = pd.to_numeric(live_states["token_id"], errors="coerce").astype(int)
    live_states["token_distance"] = pd.to_numeric(live_states["token_distance"], errors="coerce")
    live_states["regime_name"] = live_states["token_id"].map(
        lambda token: profiles.get(int(token), {}).get("regime_name") or f"token {int(token)}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"{station}_live_semantic_history.csv"
    history = _append_history(history_path, live_states)

    latest = live_states.iloc[-1]
    latest_history = history.tail(max(1, int(history_length)))
    live_token_sequence = [int(token) for token in latest_history["token_id"].tolist()]
    token_id = int(latest["token_id"])
    token_distance = float(latest["token_distance"])
    profile = profiles.get(token_id, {})
    similar = _similar_windows(embedding[-1], states=states, top_k=top_k)
    phase_candidates, phase_support, transition_source = _phase_candidates_from_sequence(
        token_sequence=live_token_sequence,
        states=states,
        profiles=profiles,
        horizon_steps=horizon_steps,
        top_k=top_k,
        min_support=min_support,
    )

    embedding_payload = {f"embedding_{idx:02d}": float(value) for idx, value in enumerate(embedding[-1])}
    feature_payload = latest.to_dict()
    missing_features = [column for column in feature_columns if column not in feature_rows.columns]

    result = {
        "station": station,
        "source_csv": str(live_csv),
        "metadata_path": str(metadata_path),
        "semantic_run_name": metadata.get("run_name"),
        "live_window": diagnostics_rows[-1],
        "live_history": {
            "history_path": str(history_path),
            "generated_windows_this_run": int(len(live_states)),
            "stored_live_windows": int(len(history)),
            "history_length_requested": int(history_length),
            "live_token_sequence": live_token_sequence,
        },
        "adapter_warning": (
            "AviationWeather METAR is usually hourly, while the historical semantic build may be based on "
            f"{metadata.get('window_size')} higher-frequency rows. Schema is matched, but sampling frequency can differ."
        ),
        "live_state": {
            "window_id": feature_payload.get("window_id"),
            "window_start": feature_payload.get("window_start"),
            "window_end": feature_payload.get("window_end"),
            "window_rows": feature_payload.get("window_rows"),
            "time_span_hours": feature_payload.get("time_span_hours"),
            "token_id": token_id,
            "token_distance": token_distance,
            "regime_name": profile.get("regime_name") or f"token {token_id}",
            "short_explanation": profile.get("short_explanation", ""),
            "meteorological_interpretation": profile.get("meteorological_interpretation", ""),
        },
        "features": {column: feature_payload.get(column) for column in feature_columns},
        "missing_features_filled_by_encoder": missing_features,
        "embedding": embedding_payload,
        "similar_historical_windows": similar,
        "phase_prediction": {
            "horizon_steps": int(horizon_steps),
            "candidate_next_phases": phase_candidates,
            "support": int(phase_support),
            "minimum_support": int(min_support),
            "support_is_sufficient": bool(phase_support >= max(1, int(min_support))),
            "transition_source": transition_source,
        },
    }

    json_path = output_dir / f"{station}_live_semantic_state.json"
    csv_path = output_dir / f"{station}_live_semantic_state.csv"
    json_path.write_text(json.dumps(_json_safe(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    flat_row = {
        **{key: result["live_state"].get(key) for key in result["live_state"]},
        **result["features"],
        **result["embedding"],
        "phase_support": phase_support,
        "top_phase_token": phase_candidates[0]["token_id"] if phase_candidates else None,
        "top_phase_probability": phase_candidates[0]["probability"] if phase_candidates else None,
    }
    pd.DataFrame([flat_row]).to_csv(csv_path, index=False)
    result["output_paths"] = {"json": str(json_path), "csv": str(csv_path), "history": str(history_path)}
    return _json_safe(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build a live semantic wind state from AviationWeather METAR rows.")
    parser.add_argument("--live_csv", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--station", default="KBOS")
    parser.add_argument("--live_window_rows", type=int, default=6)
    parser.add_argument("--live_step_rows", type=int, default=1)
    parser.add_argument("--history_length", type=int, default=4)
    parser.add_argument("--min_support", type=int, default=5)
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--phase_horizon_steps", type=int, default=1)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/live/semantic_states"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_live_semantic_state(
        live_csv=args.live_csv,
        metadata_path=args.metadata,
        station=args.station.upper(),
        live_window_rows=args.live_window_rows,
        live_step_rows=args.live_step_rows,
        history_length=args.history_length,
        min_support=args.min_support,
        backfill=args.backfill,
        top_k=args.top_k,
        horizon_steps=args.phase_horizon_steps,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
