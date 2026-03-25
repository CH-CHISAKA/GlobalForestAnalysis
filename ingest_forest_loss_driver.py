from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from dotenv import load_dotenv
from rasterio.windows import Window
from shapely.geometry import Point
from supabase import Client, create_client


# ============================================================
# Config
# ============================================================

load_dotenv()


@dataclass
class Settings:
    supabase_url: str
    supabase_key: str
    table_name: str
    tif_path: str
    countries_path: str
    dataset_name: str
    dataset_version: str
    chunk_size: int = 1024
    batch_size: int = 500
    upload_enabled: bool = True
    save_local_csv: bool = True
    output_csv_path: str = "forest_loss_output_driver/forest_loss_driver_country_summary.csv"


def load_settings() -> Settings:
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_KEY")
        or ""
    ).strip()

    upload_enabled = bool(supabase_url and supabase_key)

    return Settings(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        table_name=os.getenv("SUPABASE_TABLE", "forest_loss_driver_country"),
        tif_path=os.environ["TIF_PATH"],
        countries_path=os.environ["COUNTRIES_PATH"],
        dataset_name=os.getenv(
            "DATASET_NAME",
            "drivers_forest_loss_1km_2001_2024_v1_2",
        ),
        dataset_version=os.getenv("DATASET_VERSION", "v1_2"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1024")),
        batch_size=int(os.getenv("BATCH_SIZE", "500")),
        upload_enabled=upload_enabled,
        save_local_csv=os.getenv("SAVE_LOCAL_CSV", "true").lower() == "true",
        output_csv_path=os.getenv(
            "OUTPUT_CSV_PATH",
            "forest_loss_output_driver/forest_loss_driver_country_summary.csv",
        ),
    )


# ============================================================
# Driver mapping
# Replace with the official legend for your TIFF if available
# ============================================================

DRIVER_MAP: Dict[int, str] = {
    0: "no_data_or_unclassified",
    1: "commodity_agriculture",
    2: "shifting_agriculture",
    3: "forestry",
    4: "wildfire",
    5: "urbanization",
    6: "mining",
    7: "infrastructure",
}


# ============================================================
# Utility
# ============================================================

def chunked(items: List[dict], size: int) -> Iterable[List[dict]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def estimate_pixel_area_ha(src: rasterio.io.DatasetReader) -> float:
    """
    Estimate pixel area in hectares.

    If raster CRS is projected in meters:
        pixel_area_ha = abs(res_x * res_y) / 10000

    If raster CRS is geographic:
        falls back to 100 ha for a nominal 1km x 1km raster.
    """
    if src.crs and src.crs.is_projected:
        res_x, res_y = src.res
        return abs(res_x * res_y) / 10000.0
    return 100.0


def print_country_schema(countries: gpd.GeoDataFrame) -> None:
    print("\n[INFO] Available country columns:")
    print(countries.columns.tolist())
    preview_cols = [c for c in countries.columns if c != "geometry"][:10]
    if preview_cols:
        print("\n[INFO] Country preview:")
        print(countries[preview_cols].head(5).to_string(index=False))


def validate_country_columns(countries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Find an ISO-like column in the boundary dataset and standardize it to 'iso'.
    """
    print_country_schema(countries)

    candidates = [
        "iso",
        "ISO",
        "iso3",
        "ISO3",
        "ISO_A3",
        "ADM0_A3",
        "WB_A3",
        "GU_A3",
        "SOV_A3",
        "country_code",
        "COUNTRY_CODE",
        "GID_0",
        "ISO3166-1-Alpha-3",
        "ISO3166_1_Alpha_3",
        "ISO3166_1_ALPHA_3",
        "iso3166-1-alpha-3",
        "iso3166_1_alpha_3",
    ]

    chosen: Optional[str] = None
    for col in candidates:
        if col in countries.columns:
            chosen = col
            break

    if chosen is None:
        # fallback: case-insensitive / normalized match
        normalized_map = {
            col.lower().replace("-", "").replace("_", "").replace(" ", ""): col
            for col in countries.columns
        }

        normalized_candidates = [
            "iso",
            "iso3",
            "isoa3",
            "adm0a3",
            "wba3",
            "gua3",
            "sova3",
            "countrycode",
            "gid0",
            "iso31661alpha3",
        ]

        for n in normalized_candidates:
            if n in normalized_map:
                chosen = normalized_map[n]
                break

    if chosen is None:
        raise ValueError(
            "No ISO-like column found in country boundaries.\n"
            f"Available columns: {countries.columns.tolist()}\n\n"
            "Update validate_country_columns() to map the correct field in your file."
        )

    if chosen != "iso":
        countries = countries.rename(columns={chosen: "iso"})

    countries = countries[["iso", "geometry"]].copy()
    countries["iso"] = countries["iso"].astype(str).str.upper().str.strip()

    invalid_iso = {"-99", "NAN", "NONE", "", "NULL"}
    countries = countries[~countries["iso"].isin(invalid_iso)]
    countries = countries.dropna(subset=["iso", "geometry"])
    countries = countries[countries.geometry.notnull()]
    countries = countries[countries.geometry.is_valid]

    if countries.empty:
        raise ValueError("Country boundary file has no valid ISO + geometry rows after cleaning.")

    print(f"\n[INFO] Using ISO column: '{chosen}'")
    print(f"[INFO] Valid country rows after cleaning: {len(countries):,}")

    return countries


def create_supabase_client(settings: Settings) -> Optional[Client]:
    if not settings.upload_enabled:
        return None
    return create_client(settings.supabase_url, settings.supabase_key)


# ============================================================
# Core processing
# ============================================================

def process_raster_to_country_summary(
    tif_path: str,
    countries_path: str,
    chunk_size: int,
    pixel_area_ha: float,
    dataset_name: str,
    dataset_version: str,
) -> pd.DataFrame:
    """
    Memory-safer workflow:
    - read one raster window
    - create points for that window only
    - spatially join to countries
    - aggregate immediately
    - keep only grouped results in memory
    """
    countries = gpd.read_file(countries_path)
    countries = validate_country_columns(countries)

    if countries.crs is None:
        countries = countries.set_crs("EPSG:4326")

    all_summaries: List[pd.DataFrame] = []

    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        transform = src.transform

        print("\n[INFO] Raster metadata:")
        print(f"CRS: {src.crs}")
        print(f"Size: {src.width} x {src.height}")
        print(f"Bands: {src.count}")
        print(f"Nodata: {nodata}")
        print(f"Resolution: {src.res}")
        print(f"Estimated pixel area (ha): {pixel_area_ha}")

        total_windows = math.ceil(src.height / chunk_size) * math.ceil(src.width / chunk_size)
        current_window = 0

        for row_off in range(0, src.height, chunk_size):
            for col_off in range(0, src.width, chunk_size):
                current_window += 1
                print(f"Processing window {current_window}/{total_windows}...")

                window = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=min(chunk_size, src.width - col_off),
                    height=min(chunk_size, src.height - row_off),
                )

                arr = src.read(1, window=window)

                if nodata is None:
                    if np.issubdtype(arr.dtype, np.floating):
                        valid_mask = ~np.isnan(arr)
                    else:
                        valid_mask = np.ones_like(arr, dtype=bool)
                else:
                    valid_mask = arr != nodata

                rr, cc = np.where(valid_mask)
                if len(rr) == 0:
                    continue

                values = arr[rr, cc]

                keep = ~pd.isna(values)
                rr = rr[keep]
                cc = cc[keep]
                values = values[keep]

                if len(values) == 0:
                    continue

                global_rows = rr + row_off
                global_cols = cc + col_off

                xs, ys = rasterio.transform.xy(
                    transform,
                    global_rows,
                    global_cols,
                    offset="center",
                )

                pixels_df = pd.DataFrame(
                    {
                        "lon": np.asarray(xs, dtype=float),
                        "lat": np.asarray(ys, dtype=float),
                        "driver_code": np.asarray(values, dtype=int),
                    }
                )

                pixels_gdf = gpd.GeoDataFrame(
                    pixels_df,
                    geometry=[Point(xy) for xy in zip(pixels_df["lon"], pixels_df["lat"])],
                    crs="EPSG:4326",
                )

                local_countries = countries
                if local_countries.crs != pixels_gdf.crs:
                    local_countries = local_countries.to_crs(pixels_gdf.crs)

                joined = gpd.sjoin(
                    pixels_gdf,
                    local_countries,
                    how="inner",
                    predicate="within",
                )

                if joined.empty:
                    continue

                summary = (
                    joined.groupby(["iso", "driver_code"], as_index=False)
                    .size()
                    .rename(columns={"size": "pixel_count"})
                )

                all_summaries.append(summary)

    if not all_summaries:
        return pd.DataFrame(
            columns=[
                "iso",
                "driver_code",
                "pixel_count",
                "area_ha",
                "driver_label",
                "dataset_name",
                "dataset_version",
            ]
        )

    final_summary = pd.concat(all_summaries, ignore_index=True)
    final_summary = (
        final_summary.groupby(["iso", "driver_code"], as_index=False)["pixel_count"]
        .sum()
    )

    final_summary["driver_code"] = final_summary["driver_code"].astype(int)
    final_summary["pixel_count"] = final_summary["pixel_count"].astype("int64")
    final_summary["area_ha"] = final_summary["pixel_count"] * pixel_area_ha
    final_summary["driver_label"] = (
        final_summary["driver_code"].map(DRIVER_MAP).fillna("unknown")
    )
    final_summary["dataset_name"] = dataset_name
    final_summary["dataset_version"] = dataset_version

    final_summary = final_summary.sort_values(
        by=["iso", "driver_code"]
    ).reset_index(drop=True)

    return final_summary


# ============================================================
# Output
# ============================================================

def save_summary_csv(summary_df: pd.DataFrame, path: str) -> None:
    if summary_df.empty:
        print("[INFO] No summary rows to save locally.")
        return

    # 👇 CREATE DIRECTORY IF IT DOESN'T EXIST
    os.makedirs(os.path.dirname(path), exist_ok=True)

    summary_df.to_csv(path, index=False)
    print(f"[INFO] Saved local CSV: {path}")


def upload_summary(
    supabase: Optional[Client],
    table_name: str,
    summary_df: pd.DataFrame,
    batch_size: int,
) -> None:
    if summary_df.empty:
        print("[INFO] No rows to upload.")
        return

    if supabase is None:
        print("[INFO] Supabase credentials not found. Skipping upload.")
        return

    records = summary_df.to_dict(orient="records")
    total = len(records)
    uploaded = 0

    for batch in chunked(records, batch_size):
        response = (
            supabase.table(table_name)
            .upsert(batch, on_conflict="iso,driver_code")
            .execute()
        )
        uploaded += len(batch)
        print(f"[INFO] Uploaded {uploaded}/{total} rows")

        if getattr(response, "data", None) is None:
            print("[WARN] Upsert returned no data payload for this batch.")


# ============================================================
# Main
# ============================================================

def main() -> None:
    settings = load_settings()

    if not os.path.exists(settings.tif_path):
        raise FileNotFoundError(f"TIF file not found: {settings.tif_path}")

    if not os.path.exists(settings.countries_path):
        raise FileNotFoundError(f"Country boundary file not found: {settings.countries_path}")

    supabase = create_supabase_client(settings)

    with rasterio.open(settings.tif_path) as src:
        pixel_area_ha = estimate_pixel_area_ha(src)

    summary_df = process_raster_to_country_summary(
        tif_path=settings.tif_path,
        countries_path=settings.countries_path,
        chunk_size=settings.chunk_size,
        pixel_area_ha=pixel_area_ha,
        dataset_name=settings.dataset_name,
        dataset_version=settings.dataset_version,
    )

    print(f"\n[INFO] Summary rows: {len(summary_df):,}")
    if not summary_df.empty:
        print(summary_df.head(20).to_string(index=False))

    if settings.save_local_csv:
        save_summary_csv(summary_df, settings.output_csv_path)

    upload_summary(
        supabase=supabase,
        table_name=settings.table_name,
        summary_df=summary_df,
        batch_size=settings.batch_size,
    )

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()