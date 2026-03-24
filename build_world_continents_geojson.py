import json
from pathlib import Path
from collections import defaultdict

import geopandas as gpd

INPUT_PATH = Path("assets/maps/world_countries.geojson")
OUTPUT_PATH = Path("assets/maps/world_continents.geojson")


def resolve_continent_column(gdf: gpd.GeoDataFrame) -> str:
    candidates = [
        "continent",
        "CONTINENT",
        "region_un",
        "REGION_UN",
        "region_wb",
        "REGION_WB",
    ]
    for col in candidates:
        if col in gdf.columns:
            return col
    raise ValueError(
        f"Could not find a continent-like column. Available columns: {list(gdf.columns)}"
    )


def normalize_continent(value: str) -> str:
    text = str(value).strip()

    mapping = {
        "Africa": "Africa",
        "Asia": "Asia",
        "Europe": "Europe",
        "North America": "North America",
        "South America": "South America",
        "Oceania": "Oceania",
        "Antarctica": "Antarctica",
        "Seven seas (open ocean)": "Oceania",
    }

    return mapping.get(text, text)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    gdf = gpd.read_file(INPUT_PATH)
    continent_col = resolve_continent_column(gdf)

    gdf["continent_name"] = gdf[continent_col].apply(normalize_continent)

    gdf = gdf[gdf["continent_name"].notna()].copy()
    gdf = gdf[gdf["continent_name"] != ""].copy()

    dissolved = gdf.dissolve(by="continent_name", as_index=False)

    dissolved = dissolved[["continent_name", "geometry"]].rename(
        columns={"continent_name": "continent"}
    )

    dissolved.to_file(OUTPUT_PATH, driver="GeoJSON")
    print(f"Saved continent GeoJSON to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()