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

### 4. Configure environment

Create a `.env` file in the project root:

```env
# Optional — only needed for AI map generation features
GEMINI_API_KEY=your_key_here
```

### 5. Run the app

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
