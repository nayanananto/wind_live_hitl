# Live Wind HITL Semantic Pipeline

This package fetches live-ish AviationWeather METAR observations for a station, converts the latest wind observations into a semantic wind-state window, retrieves similar historical KBOS states, and produces a grounded next-phase suggestion.

## Local Run

```powershell
python -m pip install -r requirements.txt

python scripts\fetch_aviationweather_metar.py `
  --station KBOS `
  --hours 12 `
  --output_dir data\live\aviationweather_metar

python scripts\build_live_semantic_state.py `
  --live_csv data\live\aviationweather_metar\KBOS\metar_live.csv `
  --metadata data\metadata\kbos_5min_phase_semantic_build.json `
  --station KBOS `
  --live_window_rows 6 `
  --live_step_rows 1 `
  --history_length 4 `
  --min_support 5 `
  --backfill `
  --top_k 5 `
  --phase_horizon_steps 1 `
  --output_dir data\live\semantic_states
```

## Outputs

- `data/live/aviationweather_metar/KBOS/metar_live.csv`: raw accumulated METAR observations.
- `data/live/semantic_states/KBOS_live_semantic_state.json`: HITL-ready live semantic state with explanation fields, similar historical windows, and phase prediction.
- `data/live/semantic_states/KBOS_live_semantic_state.csv`: flattened latest live semantic state.
- `data/live/semantic_states/KBOS_live_semantic_history.csv`: accumulated rolling live semantic windows used for sequence-based phase prediction.

## GitHub Action

The workflow `.github/workflows/fetch_aviationweather_metar.yml` runs every 15 minutes and can also be triggered manually from the Actions tab.

METAR observations are usually hourly, so repeated 15-minute runs often produce no new row. The script deduplicates observations before committing.
