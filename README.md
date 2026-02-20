# 🧠 BusAI Smart Transit Planner

> AI-driven urban bus network design, simulation, and GTFS export platform — built for any city on Earth.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Module                        | Capability                                                                                     |
| ----------------------------- | ---------------------------------------------------------------------------------------------- |
| 🌍 **City Bootstrap**         | Auto-fetches bus stops from OpenStreetMap for any city; smart boundary-aware radius; CSV cache |
| 🧠 **Demand ML Ensemble**     | XGBoost + CNN (city-agnostic) + GNN → weighted ensemble demand forecast                        |
| 🛣️ **Corridor Detection**     | DBSCAN + PCA linearity → flags high-demand spines + BRT candidates                             |
| 🚌 **Trunk–Feeder Design**    | Automatically synthesises hierarchical trunk + feeder route structure                          |
| 🔁 **Transfer Hub Optimiser** | DBSCAN hub clustering + feeder endpoint snapping                                               |
| ⏰ **Temporal Scheduling**    | Peak / off-peak service tuning with frequency optimisation                                     |
| 📈 **Load Simulation**        | Stochastic Poisson hourly load curves + adaptive extra-bus dispatch                            |
| 🗺️ **Professional Map**       | Dark-matter Folium map with trunk/feeder hierarchy + hub glow + city boundary                  |
| 📦 **GTFS Export**            | Valid GTFS feed (8 files) including `frequencies.txt` + auto-validator                         |
| 🤖 **Auto-Tune**              | City-scale classifier (mega-metro → small city) auto-adjusts all service parameters            |

---

## 🗂️ Project Structure

```
AI City Bus Planner/
├── app.py                     # Main Streamlit application
├── requirements.txt
├── .env                       # API keys (not committed)
├── example.csv                # Sample stop coordinates
│
├── core/                      # Planning engine
│   ├── clustering.py
│   ├── route_optimizer.py
│   ├── bus_allocator.py
│   ├── stop_spacing_optimizer.py
│   ├── frequency_optimizer.py
│   ├── temporal_scheduler.py
│   ├── load_simulator.py
│   ├── adaptive_rerouting.py
│   ├── corridor_detector.py
│   ├── trunk_feeder.py
│   ├── transfer_hubs.py
│   ├── gtfs_exporter.py
│   └── gtfs_validator.py
│
├── ml/                        # Demand modelling
│   ├── demand_pipeline.py
│   ├── demand_model.py        # XGBoost
│   ├── deep_demand_model.py   # City-agnostic CNN (PyTorch)
│   ├── gnn_demand_model.py    # GNN
│   ├── synthetic_demand.py
│   ├── feature_engineering.py
│   └── grid_builder.py
│
├── utils/
│   ├── city_bootstrap.py      # OSM stop fetcher + CSV cache
│   ├── city_boundary.py       # Boundary polygon + clip
│   ├── city_scale.py          # Scale classifier + auto-params
│   └── map_visualizer.py
│
├── data/
│   └── city_cache/            # Cached city CSVs (auto-created)
│
└── outputs/                   # Generated routes, maps, GTFS
```

# 🧠 BusAI Planner — Key Implementations

## 🧠 Machine Learning (Demand Intelligence)

- Implemented multi-model transit demand prediction using **XGBoost, CNN, and GNN** ensemble.
- Built city-agnostic CNN demand model with dynamic grid sizing and global pooling.
- Added graph neural network (GNN) spatial refinement for neighborhood demand smoothing.
- Engineered urban features including population density, road density, and accessibility metrics.
- Created synthetic demand generator for cold-start cities without historical data.
- Implemented feature importance analytics for model interpretability.
- Added safe ML inference wrappers to prevent runtime crashes.
- Designed ensemble fusion pipeline to combine tree, deep, and graph predictions.
- Enabled automatic demand normalization across different city scales.
- Integrated model comparison diagnostics for validation during development.

## 🗺️ Core Transit Planning Engine

- Built automatic city ingestion pipeline from latitude/longitude or city name.
- Implemented dynamic city radius detection using real urban footprint.
- Added metro vs city scale detection for adaptive planning parameters.
- Developed OSM-based real road network extraction using OSMnx.
- Implemented road-following route generation (no straight-line routing).
- Built corridor detection system for BRT-style trunk identification.
- Added trunk–feeder network generation for hierarchical transit design.
- Implemented transfer hub optimization using spatial clustering.
- Built bus stop spacing optimization based on urban density.
- Added dynamic route frequency optimization using demand and capacity.
- Implemented peak vs off-peak service scheduling.
- Built passenger load simulation over 24-hour horizon.
- Added real-time adaptive re-routing simulation framework.
- Implemented bus allocation optimizer based on load factor targets.
- Added strict GTFS validation pipeline.

## 📦 GTFS Export (Deployment Ready)

- Generated full GTFS feed for real-world transit deployment.
- Implemented `shapes.txt` export for exact route geometry.
- Added frequency-based GTFS (`frequencies.txt`) support.
- Built `stop_times` and `trips` generation from simulated schedules.
- Implemented GTFS edge-case validation and repair.
- Ensured GTFS-compliant transfer hub modeling.

## 🎨 Visualization & UI (Streamlit Control Center)

- Built dark-mode transit control center UI in Streamlit.
- Implemented interactive road-following route preview map.
- Added AI demand heatmap visualization.
- Created Routes & Transfer Hub live preview.
- Built load simulation dashboard per route.
- Added automatic city caching to CSV for fast reloads.
- Implemented multi-tab analytical dashboard.
- Added professional transit color palette rendering.
- Built responsive control panel for planning parameters.
- Implemented safe caching to prevent redundant heavy computations.

## ⚙️ System Robustness & Production Readiness

- Added automatic fallback when OSM/Nominatim unavailable.
- Implemented city-agnostic processing pipeline (works globally).
- Built defensive error handling across ML and routing stack.
- Enabled high-performance caching with Streamlit `cache_data`.
- Designed modular core architecture for extensibility.
- Optimized pipeline for hackathon-speed execution.


---

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/your-username/ai-city-bus-planner.git
cd ai-city-bus-planner
```

### 2. Create virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **PyTorch note:** if the default PyTorch install doesn't match your CUDA version, visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) and install the right wheel before running the above.

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🌍 Usage

1. **Enter city name + coordinates** in the sidebar (e.g. `Mumbai`, `19.0760`, `72.8777`)
2. Click **🌍 Load City Data** — stops are fetched from OSM and cached locally
3. Adjust service parameters (buses, spacing, load factor) or leave **🤖 Auto-Tune** on
4. Click **🚀 Generate Smart Plan**
5. Explore the 6 output tabs:
   - 📊 Dashboard — executive KPIs
   - 🔥 Demand Heatmap
   - 🗺️ Routes & Transfer Hub Preview
   - 📈 Load Simulation
   - 🛣️ Corridor Analysis
   - ⬇️ Downloads (per-bus CSVs + GTFS zip)

---

## 📦 GTFS Output

The exported GTFS bundle contains:

| File              | Contents                |
| ----------------- | ----------------------- |
| `agency.txt`      | Operator metadata       |
| `stops.txt`       | All stop coordinates    |
| `routes.txt`      | Route definitions       |
| `trips.txt`       | Trip records            |
| `stop_times.txt`  | Arrival/departure times |
| `calendar.txt`    | Service calendar        |
| `shapes.txt`      | Route geometry          |
| `frequencies.txt` | Peak/off-peak headways  |

---

## 🧱 Dependencies

| Package                       | Purpose                       |
| ----------------------------- | ----------------------------- |
| `streamlit`                   | Web UI                        |
| `osmnx`                       | OSM road network + stop fetch |
| `geopandas`                   | Boundary polygon operations   |
| `folium` / `streamlit-folium` | Interactive maps              |
| `plotly`                      | Charts                        |
| `xgboost`                     | Demand regression             |
| `torch`                       | CNN demand model              |
| `torch-geometric`             | GNN demand model              |
| `scikit-learn`                | Clustering (KMeans, DBSCAN)   |
| `networkx`                    | Road-following routing        |
| `geopy`                       | Geocoding fallback            |
| `python-dotenv`               | `.env` loading                |

---

## 🤝 Contributing

Pull requests welcome. For major changes, please open an issue first.

---

## 📄 License

MIT © 2026
