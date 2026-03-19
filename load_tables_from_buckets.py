import os
import json
from typing import Any, Dict, Iterable, List, Set, Optional

from dotenv import load_dotenv
from supabase import create_client, Client

# =========================================================
# LOAD ENV
# =========================================================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")

# =========================================================
# CONFIG
# =========================================================
JSON_BUCKET = "forest-json"
STORAGE_VERSION = "v1"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

FILE_MAP = {
    "canonical_country_table": f"{STORAGE_VERSION}/canonical_country_table.json",
    "continent_watchlists": f"{STORAGE_VERSION}/continent_watchlists.json",
    "country_year_features": f"{STORAGE_VERSION}/country_year_features.json",
    "forecasts_country": f"{STORAGE_VERSION}/forecasts.json",
    "ranking_outputs": f"{STORAGE_VERSION}/ranking_outputs.json",
    "region_aggregates": f"{STORAGE_VERSION}/region_aggregates.json",
    "top_at_risk_countries": f"{STORAGE_VERSION}/top_at_risk_countries.json",
    "top_improving_countries": f"{STORAGE_VERSION}/top_improving_countries.json",
    "top_worsening_subregions": f"{STORAGE_VERSION}/top_worsening_subregions.json",
    "trend_summary_country": f"{STORAGE_VERSION}/trend_summary.json",
    "country_rankings": f"{STORAGE_VERSION}/country_rankings.json",
}

# =========================================================
# HELPERS
# =========================================================
def chunked(rows: List[Dict[str, Any]], size: int = 500) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def download_json_from_storage(bucket: str, path: str) -> Any:
    print(f"Downloading {bucket}/{path} ...")
    file_bytes = supabase.storage.from_(bucket).download(path)
    data = json.loads(file_bytes.decode("utf-8"))
    if isinstance(data, list):
        print(f"Downloaded {len(data)} rows from {bucket}/{path}")
    elif isinstance(data, dict):
        print(f"Downloaded JSON object from {bucket}/{path}")
    else:
        raise ValueError(f"{bucket}/{path} must contain a JSON array or object")
    return data


def safe_download(bucket: str, path: str, label: str) -> Optional[Any]:
    try:
        return download_json_from_storage(bucket, path)
    except Exception as e:
        print(f"Skipping {label}: {e}")
        return None


def upsert_rows(table_name: str, rows: List[Dict[str, Any]], on_conflict: str, batch_size: int = 500) -> None:
    if not rows:
        print(f"Skipping {table_name}: no rows")
        return

    total = 0
    for batch in chunked(rows, batch_size):
        supabase.table(table_name).upsert(batch, on_conflict=on_conflict).execute()
        total += len(batch)
        print(f"{table_name}: loaded {total}/{len(rows)} rows")

    print(f"Finished loading {table_name}")


def extract_iso_set(rows: List[Dict[str, Any]]) -> Set[str]:
    return {str(r["iso"]).strip().upper() for r in rows if r.get("iso")}


def filter_rows_by_valid_iso(
    rows: List[Dict[str, Any]],
    valid_isos: Set[str],
    table_name: str
) -> List[Dict[str, Any]]:
    kept = []
    dropped_isos = set()

    for r in rows:
        iso = r.get("iso")
        if iso is None:
            continue

        iso_clean = str(iso).strip().upper()
        r["iso"] = iso_clean

        if iso_clean in valid_isos:
            kept.append(r)
        else:
            dropped_isos.add(iso_clean)

    if dropped_isos:
        print(
            f"{table_name}: dropped {len(dropped_isos)} orphan ISO values "
            f"not found in canonical_country_table: {sorted(dropped_isos)}"
        )

    print(f"{table_name}: kept {len(kept)}/{len(rows)} rows after ISO validation")
    return kept


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value))
    except Exception:
        return None


def dedupe_by_keys(rows: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    seen = {}
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        seen[key] = row
    return list(seen.values())


# =========================================================
# NORMALIZERS
# =========================================================
def normalize_canonical_country_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for r in rows:
        normalized.append(
            {
                "iso": str(r["iso"]).strip().upper(),
                "country_name": r["country_name"],
                "continent": r["continent"],
                "subregion": r["subregion"],
                "extent_2000_ha": _to_float(r.get("extent_2000_ha")),
                "gain_2000_2020_ha": _to_float(r.get("gain_2000_2020_ha")),
                "loss_total_2001_2020": _to_float(r.get("loss_total_2001_2020")),
                "recovery_gap_ha": _to_float(r.get("recovery_gap_ha")),
            }
        )
    return normalized


def normalize_country_year_features(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for r in rows:
        year = _to_int(r.get("year"))
        if year is None:
            continue
        normalized.append(
            {
                "iso": str(r["iso"]).strip().upper(),
                "year": year,
                "annual_loss_ha": _to_float(r.get("annual_loss_ha")),
                "annual_primary_loss_ha": _to_float(r.get("annual_primary_loss_ha")),
                "cumulative_loss_ha": _to_float(r.get("cumulative_loss_ha")),
                "continent": r.get("continent"),
                "subregion": r.get("subregion"),
            }
        )
    return normalized


def normalize_trend_summary_country(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for r in rows:
        iso = r.get("iso")
        if not iso:
            continue
        normalized.append(
            {
                "iso": str(iso).strip().upper(),
                "loss_slope_ha_per_year": _to_float(r.get("loss_slope_ha_per_year")),
                "intercept": _to_float(r.get("intercept")),
                "r_squared": _to_float(r.get("r_squared")),
                "trend_direction": r.get("trend_direction"),
            }
        )
    return normalized


def normalize_forecasts_country(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for r in rows:
        iso = r.get("iso")
        if not iso:
            continue
        iso_clean = str(iso).strip().upper()
        grouped.setdefault(iso_clean, []).append(r)

    def best_label(labels: List[Any]) -> Any:
        priority = {
            "critical": 6,
            "urgent": 5,
            "high_risk": 4,
            "high_priority": 4,
            "watchlist": 3,
            "emerging_risk": 2,
            "stable_or_improving": 1,
            "low_risk": 0,
        }
        cleaned = [str(x) for x in labels if x is not None]
        if not cleaned:
            return None
        return max(cleaned, key=lambda x: priority.get(x, -1))

    normalized: List[Dict[str, Any]] = []

    for iso, group in grouped.items():
        latest_actual_candidates = [
            _to_float(r.get("latest_actual_annual_loss_ha"))
            for r in group
            if r.get("latest_actual_annual_loss_ha") is not None
        ]

        avg_forecast_direct = [
            _to_float(r.get("avg_forecast_annual_loss_ha"))
            for r in group
            if r.get("avg_forecast_annual_loss_ha") is not None
        ]

        max_forecast_direct = [
            _to_float(r.get("max_forecast_annual_loss_ha"))
            for r in group
            if r.get("max_forecast_annual_loss_ha") is not None
        ]

        annual_forecast_candidates = [
            _to_float(r.get("forecast_annual_loss_ha"))
            for r in group
            if r.get("forecast_annual_loss_ha") is not None
        ]

        annual_forecast_candidates = [x for x in annual_forecast_candidates if x is not None]
        avg_forecast_direct = [x for x in avg_forecast_direct if x is not None]
        max_forecast_direct = [x for x in max_forecast_direct if x is not None]
        latest_actual_candidates = [x for x in latest_actual_candidates if x is not None]

        if avg_forecast_direct:
            avg_forecast = sum(avg_forecast_direct) / len(avg_forecast_direct)
        elif annual_forecast_candidates:
            avg_forecast = sum(annual_forecast_candidates) / len(annual_forecast_candidates)
        else:
            avg_forecast = None

        if max_forecast_direct:
            max_forecast = max(max_forecast_direct)
        elif annual_forecast_candidates:
            max_forecast = max(annual_forecast_candidates)
        else:
            max_forecast = None

        labels = [r.get("forecast_risk_label") for r in group]

        normalized.append(
            {
                "iso": iso,
                "latest_actual_annual_loss_ha": max(latest_actual_candidates) if latest_actual_candidates else None,
                "avg_forecast_annual_loss_ha": avg_forecast,
                "max_forecast_annual_loss_ha": max_forecast,
                "forecast_risk_label": best_label(labels),
            }
        )

    return normalized


def normalize_country_rankings(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []

    for r in rows:
        iso = r.get("iso")
        if not iso:
            continue

        normalized.append(
            {
                "iso": str(iso).strip().upper(),
                "country_risk_score": _to_float(r.get("country_risk_score")),
                "country_improvement_score": _to_float(r.get("country_improvement_score")),
                "priority_flag": r.get("priority_flag"),
                "explanation": r.get("explanation"),
                "score_slope": _to_float(r.get("score_slope")),
                "score_latest_loss": _to_float(r.get("score_latest_loss")),
                "score_avg_forecast": _to_float(r.get("score_avg_forecast")),
                "score_max_forecast": _to_float(r.get("score_max_forecast")),
                "score_forecast_ratio": _to_float(r.get("score_forecast_ratio")),
                "score_risk_label": _to_float(r.get("score_risk_label")),
            }
        )

    return dedupe_by_keys(normalized, ["iso"])


def normalize_region_aggregates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for r in rows:
        year = _to_int(r.get("year"))
        if year is None:
            continue
        normalized.append(
            {
                "entity_level": r["aggregation_level"],
                "entity_name": r["region_name"],
                "year": year,
                "annual_loss_ha": _to_float(r.get("annual_loss_ha")),
                "annual_primary_loss_ha": _to_float(r.get("annual_primary_loss_ha")),
                "countries_count": _to_int(r.get("country_count")),
            }
        )
    return normalized


def normalize_top_at_risk_countries(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for r in rows:
        iso = r.get("iso")
        if not iso:
            continue
        normalized.append(
            {
                "iso": str(iso).strip().upper(),
                "country_name": r.get("country_name"),
                "continent": r.get("continent"),
                "subregion": r.get("subregion"),
                "latest_actual_annual_loss_ha": _to_float(r.get("latest_actual_annual_loss_ha")),
                "avg_forecast_annual_loss_ha": _to_float(r.get("avg_forecast_annual_loss_ha")),
                "max_forecast_annual_loss_ha": _to_float(r.get("max_forecast_annual_loss_ha")),
                "loss_slope_ha_per_year_forecast": _to_float(r.get("loss_slope_ha_per_year_forecast")),
                "r_squared": _to_float(r.get("r_squared")),
                "forecast_risk_label": r.get("forecast_risk_label"),
                "forecast_risk_score": _to_float(r.get("forecast_risk_score")),
                "forecast_to_latest_ratio": _to_float(r.get("forecast_to_latest_ratio")),
                "loss_slope_ha_per_year_trend": _to_float(r.get("loss_slope_ha_per_year_trend")),
                "latest_annual_loss_ha": _to_float(r.get("latest_annual_loss_ha")),
                "trend_direction": r.get("trend_direction"),
                "trend_r_squared": _to_float(r.get("trend_r_squared")),
                "loss_slope_ha_per_year_final": _to_float(r.get("loss_slope_ha_per_year_final")),
                "score_slope": _to_float(r.get("score_slope")),
                "score_latest_loss": _to_float(r.get("score_latest_loss")),
                "score_avg_forecast": _to_float(r.get("score_avg_forecast")),
                "score_max_forecast": _to_float(r.get("score_max_forecast")),
                "score_forecast_ratio": _to_float(r.get("score_forecast_ratio")),
                "score_risk_label": _to_float(r.get("score_risk_label")),
                "country_risk_score": _to_float(r.get("country_risk_score")),
                "priority_flag": r.get("priority_flag"),
                "explanation": r.get("explanation"),
            }
        )
    return dedupe_by_keys(normalized, ["iso"])


def normalize_top_improving_countries(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for r in rows:
        iso = r.get("iso")
        if not iso:
            continue
        normalized.append(
            {
                "iso": str(iso).strip().upper(),
                "country_name": r.get("country_name"),
                "continent": r.get("continent"),
                "subregion": r.get("subregion"),
                "country_improvement_score": _to_float(r.get("country_improvement_score")),
                "latest_actual_annual_loss_ha": _to_float(r.get("latest_actual_annual_loss_ha")),
                "avg_forecast_annual_loss_ha": _to_float(r.get("avg_forecast_annual_loss_ha")),
                "max_forecast_annual_loss_ha": _to_float(r.get("max_forecast_annual_loss_ha")),
                "trend_direction": r.get("trend_direction"),
                "forecast_risk_label": r.get("forecast_risk_label"),
                "priority_flag": r.get("priority_flag"),
                "explanation": r.get("explanation"),
            }
        )
    return dedupe_by_keys(normalized, ["iso"])


def normalize_top_worsening_subregions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for r in rows:
        normalized.append(
            {
                "subregion": r.get("subregion") or r.get("entity_name"),
                "continent": r.get("continent"),
                "latest_annual_loss_ha": _to_float(r.get("latest_annual_loss_ha")),
                "avg_forecast_annual_loss_ha": _to_float(r.get("avg_forecast_annual_loss_ha")),
                "max_forecast_annual_loss_ha": _to_float(r.get("max_forecast_annual_loss_ha")),
                "trend_direction": r.get("trend_direction"),
                "forecast_risk_label": r.get("forecast_risk_label"),
                "score": _to_float(r.get("score")),
                "explanation": r.get("explanation"),
            }
        )
    return dedupe_by_keys(normalized, ["subregion"])


def normalize_continent_watchlists(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for r in rows:
        normalized.append(
            {
                "continent": r.get("continent") or r.get("entity_name"),
                "latest_annual_loss_ha": _to_float(r.get("latest_annual_loss_ha")),
                "avg_forecast_annual_loss_ha": _to_float(r.get("avg_forecast_annual_loss_ha")),
                "max_forecast_annual_loss_ha": _to_float(r.get("max_forecast_annual_loss_ha")),
                "trend_direction": r.get("trend_direction"),
                "forecast_risk_label": r.get("forecast_risk_label"),
                "score": _to_float(r.get("score")),
                "priority_flag": r.get("priority_flag"),
                "explanation": r.get("explanation"),
            }
        )
    return dedupe_by_keys(normalized, ["continent"])


# =========================================================
# BUILD country_rankings FROM ranking exports
# =========================================================
def build_country_rankings_from_rank_exports(
    top_at_risk_rows: List[Dict[str, Any]],
    top_improving_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    by_iso: Dict[str, Dict[str, Any]] = {}

    for r in top_at_risk_rows:
        iso = r["iso"]
        by_iso.setdefault(
            iso,
            {
                "iso": iso,
                "country_risk_score": None,
                "country_improvement_score": None,
                "priority_flag": None,
                "explanation": None,
                "score_slope": None,
                "score_latest_loss": None,
                "score_avg_forecast": None,
                "score_max_forecast": None,
                "score_forecast_ratio": None,
                "score_risk_label": None,
            },
        )

        by_iso[iso]["country_risk_score"] = r.get("country_risk_score")
        by_iso[iso]["priority_flag"] = r.get("priority_flag")
        by_iso[iso]["explanation"] = r.get("explanation")
        by_iso[iso]["score_slope"] = r.get("score_slope")
        by_iso[iso]["score_latest_loss"] = r.get("score_latest_loss")
        by_iso[iso]["score_avg_forecast"] = r.get("score_avg_forecast")
        by_iso[iso]["score_max_forecast"] = r.get("score_max_forecast")
        by_iso[iso]["score_forecast_ratio"] = r.get("score_forecast_ratio")
        by_iso[iso]["score_risk_label"] = r.get("score_risk_label")

    for r in top_improving_rows:
        iso = r["iso"]
        by_iso.setdefault(
            iso,
            {
                "iso": iso,
                "country_risk_score": None,
                "country_improvement_score": None,
                "priority_flag": None,
                "explanation": None,
                "score_slope": None,
                "score_latest_loss": None,
                "score_avg_forecast": None,
                "score_max_forecast": None,
                "score_forecast_ratio": None,
                "score_risk_label": None,
            },
        )
        by_iso[iso]["country_improvement_score"] = r.get("country_improvement_score")
        if by_iso[iso].get("priority_flag") is None:
            by_iso[iso]["priority_flag"] = r.get("priority_flag")
        if by_iso[iso].get("explanation") is None:
            by_iso[iso]["explanation"] = r.get("explanation")

    return list(by_iso.values())


# =========================================================
# MAIN
# =========================================================
def main():
    # Core
    canonical_raw = safe_download(JSON_BUCKET, FILE_MAP["canonical_country_table"], "canonical_country_table")
    country_year_raw = safe_download(JSON_BUCKET, FILE_MAP["country_year_features"], "country_year_features")
    trend_raw = safe_download(JSON_BUCKET, FILE_MAP["trend_summary_country"], "trend_summary_country")
    forecasts_raw = safe_download(JSON_BUCKET, FILE_MAP["forecasts_country"], "forecasts_country")
    region_raw = safe_download(JSON_BUCKET, FILE_MAP["region_aggregates"], "region_aggregates")

    # Ranking-related
    country_rankings_raw = safe_download(JSON_BUCKET, FILE_MAP["country_rankings"], "country_rankings")
    ranking_outputs_raw = safe_download(JSON_BUCKET, FILE_MAP["ranking_outputs"], "ranking_outputs")
    top_at_risk_raw = safe_download(JSON_BUCKET, FILE_MAP["top_at_risk_countries"], "top_at_risk_countries")
    top_improving_raw = safe_download(JSON_BUCKET, FILE_MAP["top_improving_countries"], "top_improving_countries")
    top_worsening_subregions_raw = safe_download(JSON_BUCKET, FILE_MAP["top_worsening_subregions"], "top_worsening_subregions")
    continent_watchlists_raw = safe_download(JSON_BUCKET, FILE_MAP["continent_watchlists"], "continent_watchlists")

    if not canonical_raw:
        raise ValueError("canonical_country_table.json is required")

    canonical_rows = normalize_canonical_country_table(canonical_raw)
    country_year_rows = normalize_country_year_features(country_year_raw or [])
    trend_rows = normalize_trend_summary_country(trend_raw or [])
    forecast_rows = normalize_forecasts_country(forecasts_raw or [])
    region_rows = normalize_region_aggregates(region_raw or [])

    # Ranking fallback resolution
    if ranking_outputs_raw and isinstance(ranking_outputs_raw, dict):
        if top_at_risk_raw is None:
            top_at_risk_raw = ranking_outputs_raw.get("top_at_risk_countries", [])
        if top_improving_raw is None:
            top_improving_raw = ranking_outputs_raw.get("top_improving_countries", [])
        if top_worsening_subregions_raw is None:
            top_worsening_subregions_raw = ranking_outputs_raw.get("top_worsening_subregions", [])
        if continent_watchlists_raw is None:
            continent_watchlists_raw = ranking_outputs_raw.get("continent_watchlists", [])

    top_at_risk_rows = normalize_top_at_risk_countries(top_at_risk_raw or [])
    top_improving_rows = normalize_top_improving_countries(top_improving_raw or [])
    top_worsening_subregions_rows = normalize_top_worsening_subregions(top_worsening_subregions_raw or [])
    continent_watchlists_rows = normalize_continent_watchlists(continent_watchlists_raw or [])

    if country_rankings_raw and isinstance(country_rankings_raw, list):
        ranking_rows = normalize_country_rankings(country_rankings_raw)
    else:
        ranking_rows = build_country_rankings_from_rank_exports(
            top_at_risk_rows=top_at_risk_rows,
            top_improving_rows=top_improving_rows,
        )
        ranking_rows = normalize_country_rankings(ranking_rows)
        if ranking_rows:
            print("Built country_rankings from ranking exports")

    valid_isos = extract_iso_set(canonical_rows)
    print(f"canonical_country_table valid ISO count: {len(valid_isos)}")

    country_year_rows = filter_rows_by_valid_iso(country_year_rows, valid_isos, "country_year_features")
    trend_rows = filter_rows_by_valid_iso(trend_rows, valid_isos, "trend_summary_country")
    forecast_rows = filter_rows_by_valid_iso(forecast_rows, valid_isos, "forecasts_country")
    ranking_rows = filter_rows_by_valid_iso(ranking_rows, valid_isos, "country_rankings")
    top_at_risk_rows = filter_rows_by_valid_iso(top_at_risk_rows, valid_isos, "top_at_risk_countries")
    top_improving_rows = filter_rows_by_valid_iso(top_improving_rows, valid_isos, "top_improving_countries")

    # Load in dependency order
    upsert_rows("canonical_country_table", canonical_rows, on_conflict="iso")
    upsert_rows("country_year_features", country_year_rows, on_conflict="iso,year")
    upsert_rows("trend_summary_country", trend_rows, on_conflict="iso")
    upsert_rows("forecasts_country", forecast_rows, on_conflict="iso")
    upsert_rows("country_rankings", ranking_rows, on_conflict="iso")
    upsert_rows("region_aggregates", region_rows, on_conflict="entity_level,entity_name,year")

    # Export/ranking support tables
    upsert_rows("top_at_risk_countries", top_at_risk_rows, on_conflict="iso")
    upsert_rows("top_improving_countries", top_improving_rows, on_conflict="iso")
    upsert_rows("top_worsening_subregions", top_worsening_subregions_rows, on_conflict="subregion")
    upsert_rows("continent_watchlists", continent_watchlists_rows, on_conflict="continent")

    print("Done loading Supabase tables.")


if __name__ == "__main__":
    main()