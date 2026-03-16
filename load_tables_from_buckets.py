import os
import json
from typing import Any, Dict, Iterable, List, Set

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
    "country_year_features": f"{STORAGE_VERSION}/country_year_features.json",
    "trend_summary_country": f"{STORAGE_VERSION}/trend_summary.json",
    "forecasts_country": f"{STORAGE_VERSION}/forecasts.json",
    "country_rankings": f"{STORAGE_VERSION}/country_rankings.json",
    "region_aggregates": f"{STORAGE_VERSION}/region_aggregates.json",
}

# =========================================================
# HELPERS
# =========================================================
def chunked(rows: List[Dict[str, Any]], size: int = 500) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def download_json_from_storage(bucket: str, path: str) -> List[Dict[str, Any]]:
    print(f"Downloading {bucket}/{path} ...")
    file_bytes = supabase.storage.from_(bucket).download(path)
    data = json.loads(file_bytes.decode("utf-8"))

    if not isinstance(data, list):
        raise ValueError(f"{bucket}/{path} must contain a JSON array")

    print(f"Downloaded {len(data)} rows from {bucket}/{path}")
    return data


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
        dropped_sorted = sorted(dropped_isos)
        print(
            f"{table_name}: dropped {len(dropped_sorted)} orphan ISO values "
            f"not found in canonical_country_table: {dropped_sorted}"
        )

    print(f"{table_name}: kept {len(kept)}/{len(rows)} rows after ISO validation")
    return kept


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
                "extent_2000_ha": r.get("extent_2000_ha"),
                "gain_2000_2020_ha": r.get("gain_2000_2020_ha"),
                "loss_total_2001_2020": r.get("loss_total_2001_2020"),
                "recovery_gap_ha": r.get("recovery_gap_ha"),
            }
        )
    return normalized


def normalize_country_year_features(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for r in rows:
        normalized.append(
            {
                "iso": str(r["iso"]).strip().upper(),
                "year": int(r["year"]),
                "annual_loss_ha": r.get("annual_loss_ha"),
                "annual_primary_loss_ha": r.get("annual_primary_loss_ha"),
                "cumulative_loss_ha": r.get("cumulative_loss_ha"),
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
                "loss_slope_ha_per_year": r.get("loss_slope_ha_per_year"),
                "intercept": r.get("intercept"),
                "r_squared": r.get("r_squared"),
                "trend_direction": r.get("trend_direction"),
            }
        )

    return normalized


def normalize_forecasts_country(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Supports:
    1. one-row-per-iso summary forecast export
    2. multi-row-per-iso yearly forecast export

    If multiple rows exist per iso, collapses to one row per iso.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for r in rows:
        iso = r.get("iso")
        if not iso:
            continue
        iso_clean = str(iso).strip().upper()
        grouped.setdefault(iso_clean, []).append(r)

    def best_label(labels: List[Any]) -> Any:
        priority = {
            "critical": 5,
            "high_priority": 4,
            "watchlist": 3,
            "medium": 2,
            "low": 1,
        }
        cleaned = [str(x) for x in labels if x is not None]
        if not cleaned:
            return None
        return max(cleaned, key=lambda x: priority.get(x, 0))

    normalized: List[Dict[str, Any]] = []

    for iso, group in grouped.items():
        latest_actual_candidates = [
            r.get("latest_actual_annual_loss_ha")
            for r in group
            if r.get("latest_actual_annual_loss_ha") is not None
        ]

        avg_forecast_direct = [
            r.get("avg_forecast_annual_loss_ha")
            for r in group
            if r.get("avg_forecast_annual_loss_ha") is not None
        ]

        max_forecast_direct = [
            r.get("max_forecast_annual_loss_ha")
            for r in group
            if r.get("max_forecast_annual_loss_ha") is not None
        ]

        annual_forecast_candidates = [
            r.get("forecast_annual_loss_ha")
            for r in group
            if r.get("forecast_annual_loss_ha") is not None
        ]

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
                "country_risk_score": r.get("country_risk_score"),
                "country_improvement_score": r.get("country_improvement_score"),
                "priority_flag": r.get("priority_flag"),
                "explanation": r.get("explanation"),
                "score_slope": r.get("score_slope"),
                "score_latest_loss": r.get("score_latest_loss"),
                "score_avg_forecast": r.get("score_avg_forecast"),
                "score_max_forecast": r.get("score_max_forecast"),
                "score_forecast_ratio": r.get("score_forecast_ratio"),
                "score_risk_label": r.get("score_risk_label"),
            }
        )

    return normalized


def normalize_region_aggregates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "entity_level": r["aggregation_level"],
            "entity_name": r["region_name"],
            "year": int(r["year"]),
            "annual_loss_ha": r.get("annual_loss_ha"),
            "annual_primary_loss_ha": r.get("annual_primary_loss_ha"),
            "countries_count": r.get("country_count"),
        }
        for r in rows
    ]


# =========================================================
# MAIN
# =========================================================
def main():
    canonical_rows = normalize_canonical_country_table(
        download_json_from_storage(JSON_BUCKET, FILE_MAP["canonical_country_table"])
    )

    country_year_rows = normalize_country_year_features(
        download_json_from_storage(JSON_BUCKET, FILE_MAP["country_year_features"])
    )

    trend_rows = normalize_trend_summary_country(
        download_json_from_storage(JSON_BUCKET, FILE_MAP["trend_summary_country"])
    )

    forecast_rows = normalize_forecasts_country(
        download_json_from_storage(JSON_BUCKET, FILE_MAP["forecasts_country"])
    )

    region_rows = normalize_region_aggregates(
        download_json_from_storage(JSON_BUCKET, FILE_MAP["region_aggregates"])
    )

    ranking_rows: List[Dict[str, Any]] = []
    try:
        ranking_rows = normalize_country_rankings(
            download_json_from_storage(JSON_BUCKET, FILE_MAP["country_rankings"])
        )
    except Exception as e:
        print(f"Skipping country_rankings for now: {e}")

    valid_isos = extract_iso_set(canonical_rows)
    print(f"canonical_country_table valid ISO count: {len(valid_isos)}")

    country_year_rows = filter_rows_by_valid_iso(country_year_rows, valid_isos, "country_year_features")
    trend_rows = filter_rows_by_valid_iso(trend_rows, valid_isos, "trend_summary_country")
    forecast_rows = filter_rows_by_valid_iso(forecast_rows, valid_isos, "forecasts_country")
    if ranking_rows:
        ranking_rows = filter_rows_by_valid_iso(ranking_rows, valid_isos, "country_rankings")

    # Load in dependency order
    upsert_rows("canonical_country_table", canonical_rows, on_conflict="iso")
    upsert_rows("country_year_features", country_year_rows, on_conflict="iso,year")
    upsert_rows("trend_summary_country", trend_rows, on_conflict="iso")
    upsert_rows("forecasts_country", forecast_rows, on_conflict="iso")

    if ranking_rows:
        upsert_rows("country_rankings", ranking_rows, on_conflict="iso")

    upsert_rows("region_aggregates", region_rows, on_conflict="entity_level,entity_name,year")

    print("Done loading Supabase tables.")


if __name__ == "__main__":
    main()