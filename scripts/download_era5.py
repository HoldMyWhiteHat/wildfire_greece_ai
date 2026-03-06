import cdsapi

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "total_precipitation",
        ],
        "year": ["2023","2024", "2025"],
        "month": ["06", "07", "08"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": "12:00",
        "area": [41, 19, 34, 28],  # Greece bounding box
        "format": "netcdf",
    },
    "data/era5_greece.nc",
)

print("ERA5 data downloaded")