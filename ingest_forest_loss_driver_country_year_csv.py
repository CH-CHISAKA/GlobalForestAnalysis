from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()


# ============================================================
# Config
# ============================================================

@dataclass
class Settings:
    supabase_url: str
    supabase_key: str
    table_name: str
    csv_path: str
    metadata_path: str
    batch_size: int = 500
    upload_enabled: bool = True
    save_local_csv: bool = True
    output_csv_path: str = "forest_loss_driver_country_year_long.csv"
    dataset_name: str = "tree-cover-loss-by-dominant-driver"
    dataset_version: str = "2025-08-26"


def load_settings() -> Settings:
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_KEY")
        or ""
    ).strip()

    return Settings(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        table_name=os.getenv("SUPABASE_TABLE", "forest_loss_driver_country_year"),
        csv_path=os.environ["CSV_PATH"],
        metadata_path=os.environ["METADATA_PATH"],
        batch_size=int(os.getenv("BATCH_SIZE", "500")),
        upload_enabled=bool(supabase_url and supabase_key),
        save_local_csv=os.getenv("SAVE_LOCAL_CSV", "true").lower() == "true",
        output_csv_path=os.getenv(
            "OUTPUT_CSV_PATH",
            "forest_loss_driver_country_year_long.csv",
        ),
        dataset_name=os.getenv("DATASET_NAME", "tree-cover-loss-by-dominant-driver"),
        dataset_version=os.getenv("DATASET_VERSION", "2025-08-26"),
    )


# ============================================================
# Driver mapping from the CSV / metadata
# ============================================================

DRIVER_COLUMN_MAP: Dict[str, Dict[str, object]] = {
    "Permanent agriculture": {
        "driver_code": 1,
        "driver_label": "permanent_agriculture",
    },
    "Shifting cultivation": {
        "driver_code": 2,
        "driver_label": "shifting_cultivation",
    },
    "Logging": {
        "driver_code": 3,
        "driver_label": "logging",
    },
    "Wildfire": {
        "driver_code": 4,
        "driver_label": "wildfire",
    },
    "Settlements and infrastructure": {
        "driver_code": 5,
        "driver_label": "settlements_and_infrastructure",
    },
    "Mining and energy industry": {
        "driver_code": 6,
        "driver_label": "mining_and_energy_industry",
    },
    "Other natural disturbances": {
        "driver_code": 7,
        "driver_label": "other_natural_disturbances",
    },
}


# ============================================================
# Utility
# ============================================================

def chunked(items: List[dict], size: int) -> Iterable[List[dict]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def create_supabase_client(settings: Settings) -> Optional[Client]:
    if not settings.upload_enabled:
        return None
    return create_client(settings.supabase_url, settings.supabase_key)


# ============================================================
# Validation
# ============================================================

def validate_input_columns(df: pd.DataFrame) -> None:
    required_base = {"Entity", "Code", "Year"}
    missing_base = required_base - set(df.columns)
    if missing_base:
        raise ValueError(f"Missing required base columns: {sorted(missing_base)}")

    missing_driver_cols = set(DRIVER_COLUMN_MAP.keys()) - set(df.columns)
    if missing_driver_cols:
        raise ValueError(
            f"Missing expected driver columns: {sorted(missing_driver_cols)}"
        )


def clean_iso(code: object) -> Optional[str]:
    if pd.isna(code):
        return None
    value = str(code).strip().upper()
    if value in {"", "OWID_WRL", "OWID_AFR", "OWID_ASI", "OWID_EUR", "OWID_NAM", "OWID_SAM", "OWID_OCE"}:
        return None
    if len(value) != 3:
        return None
    return value


# ============================================================
# Transform
# ============================================================

def transform_csv_to_long(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    validate_input_columns(df)

    working = df.copy()

    working["iso"] = working["Code"].apply(clean_iso)
    working["country_name"] = working["Entity"].astype(str).str.strip()
    working["loss_year"] = pd.to_numeric(working["Year"], errors="coerce").astype("Int64")

    working = working.dropna(subset=["iso", "loss_year"])
    working["loss_year"] = working["loss_year"].astype(int)

    driver_cols = list(DRIVER_COLUMN_MAP.keys())

    long_df = working.melt(
        id_vars=["iso", "country_name", "loss_year"],
        value_vars=driver_cols,
        var_name="driver_column",
        value_name="loss_area_ha",
    )

    long_df["loss_area_ha"] = pd.to_numeric(long_df["loss_area_ha"], errors="coerce")
    long_df = long_df.dropna(subset=["loss_area_ha"])

    long_df["driver_code"] = long_df["driver_column"].map(
        lambda x: DRIVER_COLUMN_MAP[x]["driver_code"]
    )
    long_df["driver_label"] = long_df["driver_column"].map(
        lambda x: DRIVER_COLUMN_MAP[x]["driver_label"]
    )

    long_df["dataset_name"] = settings.dataset_name
    long_df["dataset_version"] = settings.dataset_version
    long_df["source_note"] = "Global Forest Watch dominant-driver annual series via OWID export"

    long_df = long_df[
        [
            "iso",
            "country_name",
            "loss_year",
            "driver_code",
            "driver_label",
            "loss_area_ha",
            "dataset_name",
            "dataset_version",
            "source_note",
        ]
    ].copy()

    # Keep zeros if you want complete annual series.
    long_df["loss_area_ha"] = long_df["loss_area_ha"].astype(float)

    # Aggregate in case the source ever contains duplicate iso-year rows
    long_df = (
        long_df.groupby(
            [
                "iso",
                "country_name",
                "loss_year",
                "driver_code",
                "driver_label",
                "dataset_name",
                "dataset_version",
                "source_note",
            ],
            as_index=False,
        )["loss_area_ha"]
        .sum()
    )

    long_df = long_df.sort_values(
        by=["iso", "loss_year", "driver_code"]
    ).reset_index(drop=True)

    return long_df


# ============================================================
# Output
# ============================================================
def save_long_csv(df: pd.DataFrame, path: str) -> None:
    if df.empty:
        print("[INFO] No rows to save locally.")
        return

    # 👇 THIS LINE FIXES EVERYTHING
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_csv(path, index=False)
    print(f"[INFO] Saved local CSV: {path}")

def upload_to_supabase(
    supabase: Optional[Client],
    table_name: str,
    df: pd.DataFrame,
    batch_size: int,
) -> None:
    if df.empty:
        print("[INFO] No rows to upload.")
        return

    if supabase is None:
        print("[INFO] Supabase credentials not found. Skipping upload.")
        return

    records = df.to_dict(orient="records")
    total = len(records)
    uploaded = 0

    for batch in chunked(records, batch_size):
        response = (
            supabase.table(table_name)
            .upsert(batch, on_conflict="iso,loss_year,driver_code")
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

    if not os.path.exists(settings.csv_path):
        raise FileNotFoundError(f"CSV file not found: {settings.csv_path}")

    if not os.path.exists(settings.metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {settings.metadata_path}")

    supabase = create_supabase_client(settings)

    df = pd.read_csv(settings.csv_path)

    print("[INFO] Input shape:", df.shape)
    print("[INFO] Input columns:", df.columns.tolist())
    print("\n[INFO] Input preview:")
    print(df.head(5).to_string(index=False))

    long_df = transform_csv_to_long(df, settings)

    print(f"\n[INFO] Output rows: {len(long_df):,}")
    print("[INFO] Output preview:")
    print(long_df.head(20).to_string(index=False))

    if settings.save_local_csv:
        save_long_csv(long_df, settings.output_csv_path)

    upload_to_supabase(
        supabase=supabase,
        table_name=settings.table_name,
        df=long_df,
        batch_size=settings.batch_size,
    )

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()