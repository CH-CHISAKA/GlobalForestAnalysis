#!/usr/bin/env python3
from __future__ import annotations
from dotenv import load_dotenv

import os
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
from supabase import Client, create_client

# Load .env automatically
load_dotenv()

# =========================================================
# CONFIG
# =========================================================
CSV_PATH = os.getenv("CSV_PATH", "tree_cover_loss_by_driver.csv")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

DIM_TABLE = "driver_types_dim"
FACT_TABLE = "global_yearly_driver"
BATCH_SIZE = 500


# =========================================================
# DATA MODELS
# =========================================================
@dataclass
class PipelineStats:
    raw_rows: int
    cleaned_rows: int
    dim_rows: int
    fact_rows: int


# =========================================================
# HELPERS
# =========================================================
def require_env() -> None:
    if not SUPABASE_URL:
        raise ValueError("Missing SUPABASE_URL environment variable.")
    if not SUPABASE_SERVICE_ROLE_KEY:
        raise ValueError("Missing SUPABASE_SERVICE_ROLE_KEY environment variable.")


def get_supabase() -> Client:
    require_env()
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def chunk_records(records: List[Dict[str, Any]], batch_size: int = BATCH_SIZE):
    for i in range(0, len(records), batch_size):
        yield records[i:i + batch_size]


# =========================================================
# EXTRACT + TRANSFORM
# =========================================================
def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    expected = {
        "drivers_type",
        "loss_year",
        "loss_area_ha",
        "gross_carbon_emissions_Mg",
    }

    missing = expected - set(df.columns)
    if missing:
        raise KeyError(f"CSV is missing required columns: {sorted(missing)}")

    return df


def clean_global_driver_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["drivers_type"] = out["drivers_type"].astype(str).str.strip()
    out["loss_year"] = pd.to_numeric(out["loss_year"], errors="coerce").astype("Int64")
    out["loss_area_ha"] = pd.to_numeric(out["loss_area_ha"], errors="coerce")
    out["gross_carbon_emissions_Mg"] = pd.to_numeric(
        out["gross_carbon_emissions_Mg"], errors="coerce"
    )

    out = out.dropna(
        subset=[
            "drivers_type",
            "loss_year",
            "loss_area_ha",
            "gross_carbon_emissions_Mg",
        ]
    ).copy()

    out = out[out["drivers_type"] != ""].copy()
    out["loss_year"] = out["loss_year"].astype(int)

    # Ensure one record per driver_type + year
    out = (
        out.groupby(["drivers_type", "loss_year"], as_index=False)
        .agg(
            {
                "loss_area_ha": "sum",
                "gross_carbon_emissions_Mg": "sum",
            }
        )
        .sort_values(["drivers_type", "loss_year"])
        .reset_index(drop=True)
    )

    return out


# =========================================================
# BUILD TABLE PAYLOADS
# =========================================================
def build_driver_dim(df: pd.DataFrame) -> pd.DataFrame:
    dim = (
        df[["drivers_type"]]
        .drop_duplicates()
        .sort_values("drivers_type")
        .reset_index(drop=True)
        .copy()
    )

    dim["driver_key"] = range(1, len(dim) + 1)
    dim = dim[["driver_key", "drivers_type"]]
    return dim


def build_driver_fact(clean_df: pd.DataFrame, dim_df: pd.DataFrame) -> pd.DataFrame:
    fact = clean_df.merge(dim_df, on="drivers_type", how="left")

    if fact["driver_key"].isna().any():
        raise ValueError("Some drivers_type values could not be mapped to driver_key.")

    fact = fact[
        [
            "driver_key",
            "drivers_type",
            "loss_year",
            "loss_area_ha",
            "gross_carbon_emissions_Mg",
        ]
    ].copy()

    fact["record_key"] = (
        fact["drivers_type"]
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
        + "_"
        + fact["loss_year"].astype(str)
    )

    fact = fact[
        [
            "record_key",
            "driver_key",
            "drivers_type",
            "loss_year",
            "loss_area_ha",
            "gross_carbon_emissions_Mg",
        ]
    ]

    return fact.sort_values(["driver_key", "loss_year"]).reset_index(drop=True)


# =========================================================
# LOAD TO SUPABASE
# =========================================================
def upsert_dataframe(
    supabase: Client,
    table_name: str,
    df: pd.DataFrame,
    on_conflict: str,
) -> None:
    records = df.to_dict(orient="records")

    if not records:
        print(f"No records to upload for {table_name}.")
        return

    for batch in chunk_records(records):
        response = (
            supabase.table(table_name)
            .upsert(batch, on_conflict=on_conflict)
            .execute()
        )

        if getattr(response, "data", None) is None:
            raise RuntimeError(f"Upsert failed for table {table_name}.")


# =========================================================
# MAIN
# =========================================================
def run(csv_path: str = CSV_PATH) -> PipelineStats:
    print(f"Loading CSV: {csv_path}")
    raw = load_csv(csv_path)
    clean = clean_global_driver_df(raw)

    dim = build_driver_dim(clean)
    fact = build_driver_fact(clean, dim)

    print("Connecting to Supabase...")
    supabase = get_supabase()

    print(f"Uploading dimension table: {DIM_TABLE}")
    upsert_dataframe(supabase, DIM_TABLE, dim, on_conflict="drivers_type")

    print(f"Uploading fact table: {FACT_TABLE}")
    upsert_dataframe(supabase, FACT_TABLE, fact, on_conflict="record_key")

    stats = PipelineStats(
        raw_rows=len(raw),
        cleaned_rows=len(clean),
        dim_rows=len(dim),
        fact_rows=len(fact),
    )

    print("Done.")
    print(stats)
    return stats


if __name__ == "__main__":
    run()