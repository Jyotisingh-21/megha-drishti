# Extreme Event Replays

Due to data retention policies on upstream data sources (ECMWF Open Data via AWS PDS, NOAA GFS, etc.), only events occurring within the rolling archive window can be replayed.

## Wayanad Landslides (July 2024)
- **Status**: Data no longer available on ECMWF Open Data AWS mirror (retention ~40-60 days). Could not fetch.

## Cyclone Remal (May 2024)
- **Status**: Data expired from open public mirrors. Could not fetch.

## Delhi Heatwave (May 2024)
- **Status**: Data expired from open public mirrors. Could not fetch.

## Cyclone Biparjoy (June 2023)
- **Status**: Occurred prior to the availability of the current ECMWF 0.25° Open Data pipeline and AIFS models. Could not fetch.

### Note on Historical Replays
To successfully replay historical events older than 60 days, researchers must connect the pipeline to MARS (ECMWF's tape archive) using an institutional `cdsapi` token or similar, as the free public tier drops granular hourly forecasts after a few weeks.
