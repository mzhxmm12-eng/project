# HK Urban Renewal Index App

## How to run

Run the Streamlit app from the project root:

```bash
streamlit run app/streamlit_app.py
```

You can also use the helper script:

```bash
./run_app.sh
```

## Sidebar sliders

- `Skeleton (Building)`: adjusts how strongly building floor area influences the renewal priority score.
- `Metabolism (Services)`: adjusts the influence of supermarket and green space accessibility.
- `Circulatory (Roads)`: adjusts the influence of road density and transit accessibility.

The three weights should sum to `1.0`. When they do, the app recalculates the composite renewal priority score live on the map and in the ranking table.

## Data sources used

- OpenStreetMap contributors: roads, buildings, POIs, green space
- HK Government Open Data: district/block boundary data
- Thesis context: MSc Urban Informatics, The Hong Kong Polytechnic University (PolyU), 2025
