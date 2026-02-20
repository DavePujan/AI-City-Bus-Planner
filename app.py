import streamlit as st
import pandas as pd
import time
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from core.data_loader import load_stops
from core.image_stop_detector import detect_stops_from_image
from core.clustering import cluster_stops
from core.route_optimizer import optimize_routes
from core.bus_allocator import allocate_buses
from core.metrics import calculate_metrics
from utils.map_visualizer import create_bus_maps
from ml.demand_pipeline import run_demand_pipeline
from utils.osmnx_map_renderer import generate_beautiful_map
from streamlit_folium import st_folium
import folium
import plotly.express as px
from geopy.geocoders import Nominatim, ArcGIS
import networkx as nx
import osmnx as ox
from core.stop_spacing_optimizer import optimize_stop_spacing
from core.frequency_optimizer import optimize_route_frequency
from core.temporal_scheduler import apply_peak_offpeak_scheduling
from core.gtfs_exporter import export_gtfs
from core.gtfs_validator import validate_and_fix_gtfs
from core.load_simulator import simulate_passenger_load
from core.adaptive_rerouting import simulate_adaptive_rerouting
from core.corridor_detector import detect_demand_corridors
from core.trunk_feeder import generate_trunk_feeder_network
from core.transfer_hubs import optimize_transfer_hubs
from utils.city_boundary import get_city_boundary, clip_points_to_boundary
from utils.city_scale import compute_city_metrics, classify_city_scale, get_auto_parameters
import plotly.graph_objects as go
import time

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="BusAI Planner",
    page_icon="🚌",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 800;
    background: linear-gradient(90deg, #ff4b4b, #6c63ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.stMetric {
    background-color: #111318;
    padding: 14px 16px;
    border-radius: 12px;
    border: 1px solid #2a2d3a;
}
.block-container { padding-top: 1rem; }
h1, h2, h3 { letter-spacing: 0.3px; }
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown(
    "<h1 class='main-title'>🧠 BusAI Smart Transit Planner</h1>"
    "<p style='color:#9aa0a6;margin-top:0;'>"
    "AI-driven urban bus network design &amp; simulation platform</p>",
    unsafe_allow_html=True,
)

# ================= UI HELPERS =================
def show_pipeline_progress():
    """Staged progress bar for a premium feel."""
    steps = [
        "Loading city boundary...",
        "Running demand models...",
        "Designing routes...",
        "Optimizing network...",
        "Rendering maps...",
    ]
    bar = st.progress(0, text=steps[0])
    for i in range(1, 101):
        milestone = {15: steps[1], 40: steps[2], 65: steps[3], 85: steps[4]}
        if i in milestone:
            bar.progress(i, text=milestone[i])
        else:
            bar.progress(i)
        time.sleep(0.002)
    bar.empty()


def render_kpi_row(metrics, rmse, city_scale, hubs_df, trunk_routes, events_df):
    """Single-row executive KPI band shown at top of Dashboard tab."""
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("🚌 Routes",        metrics.get("total_routes", 0))
    c2.metric("🚏 Stops",         metrics.get("total_stops",  0))
    c3.metric("📉 RMSE",          f"{rmse:.1f}")
    c4.metric("🏙 City Scale",   city_scale.replace("_", " ").title())
    c5.metric("🔁 Hubs",          len(hubs_df))
    c6.metric("🚨 Interventions", len(events_df))


# ================= CACHED PIPELINE =================
@st.cache_data(show_spinner=False)
def cached_pipeline(df_tuple, num_buses, operating_hours,
                    min_spacing_m=400, bus_capacity=60, target_load_factor=0.75,
                    peak_multiplier=1.6, offpeak_multiplier=0.6, peak_share=0.35,
                    n_clusters_override=None):
    """Cached end-to-end ML + routing pipeline. Prevents recomputation on widget changes."""
    import pandas as pd
    df = pd.DataFrame(df_tuple[1], columns=df_tuple[0])

    demand_df, rmse, importance_df = run_demand_pipeline(df)

    # Corridor detection (runs on demand grid)
    corridors_df = detect_demand_corridors(demand_df)

    # Trunk-feeder structure
    trunk_routes, feeder_routes, role_df = generate_trunk_feeder_network(
        df, demand_df, corridors_df
    )

    # Better cluster count: fewer, denser clusters vs. spaghetti many-tiny
    n_clusters = n_clusters_override or max(3, min(num_buses // 2, len(df) // 6))

    # Pre-clustering: spatially deduplicate stops
    from math import radians, cos, sin, asin, sqrt
    def _haversine_m(lat1, lon1, lat2, lon2):
        R = 6371000
        dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
        return 2 * R * asin(sqrt(a))

    kept = []
    for idx, row in df.iterrows():
        if not kept:
            kept.append(idx); continue
        last = df.loc[kept[-1]]
        if _haversine_m(last["lat"], last["lon"], row["lat"], row["lon"]) >= min_spacing_m * 0.5:
            kept.append(idx)
    df_spaced = df.loc[kept].reset_index(drop=True)

    # Include trunk/feeder stops if present, else fall back to cluster routing
    if trunk_routes or feeder_routes:
        # Transfer hub optimization: snap feeder endpoints to trunk hubs
        hubs_df, feeder_routes = optimize_transfer_hubs(
            df_spaced, trunk_routes, feeder_routes
        )
        routes = {**trunk_routes, **feeder_routes}
    else:
        hubs_df = pd.DataFrame()
        clustered = cluster_stops(df_spaced, n_clusters)
        routes = optimize_routes(clustered)

    # keep clustered for metrics
    if "clustered" not in dir():
        clustered = df_spaced

    # Post-route stop spacing (removes still-redundant stops within routes)
    routes = optimize_stop_spacing(routes, demand_df=demand_df, min_spacing_m=min_spacing_m)

    allocation = allocate_buses(routes, num_buses, operating_hours, demand_df=demand_df)

    # Dynamic frequency optimization
    allocation = optimize_route_frequency(
        routes, allocation, demand_df, operating_hours,
        bus_capacity=bus_capacity, target_load_factor=target_load_factor
    )

    # Peak / off-peak scheduling
    allocation = apply_peak_offpeak_scheduling(
        routes, allocation, demand_df, operating_hours,
        peak_multiplier=peak_multiplier,
        offpeak_multiplier=offpeak_multiplier,
        peak_share=peak_share
    )

    # Hourly load simulation
    load_df = simulate_passenger_load(
        routes, allocation, demand_df, operating_hours,
        bus_capacity=bus_capacity
    )

    # Adaptive re-routing: detect overload + model extra-bus dispatch
    events_df, load_df_updated = simulate_adaptive_rerouting(
        load_df, allocation, bus_capacity=bus_capacity
    )

    metrics = calculate_metrics(clustered, routes, allocation)

    return (
        demand_df, rmse, importance_df, clustered, routes,
        allocation, metrics, load_df_updated, events_df,
        corridors_df, hubs_df
    )


@st.cache_data(show_spinner=False)
def cached_beautiful_map(routes_repr, num_buses, base_lat, base_lon):
    """Cached OSMnx map renderer to avoid re-downloading road network."""
    return generate_beautiful_map(
        routes=routes_repr,
        num_buses=num_buses,
        base_lat=base_lat,
        base_lon=base_lon
    )

@st.cache_resource(show_spinner=False)
def cached_city_boundary(city_name, lat, lon):
    """Cached city boundary polygon fetch — avoids repeated geocoding."""
    return get_city_boundary(
        city_name=city_name if city_name else None,
        lat=lat, lon=lon
    )

@st.cache_resource(show_spinner=False)
def get_osm_graph(lat, lon):
    """Cached OSM road graph for road-following preview map."""
    return ox.graph_from_point((lat, lon), dist=6000, network_type="drive")


def get_road_path(G, route):
    """
    Convert stop sequence -> road-following polyline coords.
    Falls back to empty list on failure.
    """
    if len(route) < 2:
        return []
    try:
        route_nodes = [
            ox.distance.nearest_nodes(G, X=lon, Y=lat)
            for lat, lon, _ in route
        ]
        full_path = []
        for i in range(len(route_nodes) - 1):
            try:
                segment = nx.shortest_path(
                    G, route_nodes[i], route_nodes[i + 1], weight="length"
                )
                full_path.extend(segment if not full_path else segment[1:])
            except nx.NetworkXNoPath:
                continue
        return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in full_path]
    except Exception:
        return []


def render_routes_and_hubs(routes, stops_df, hubs_df, boundary_gdf=None):
    """
    Professional transit control map.
    Dark-matter tiles, trunk/feeder visual hierarchy, hub glow effect.
    Routes must be dicts of {route_id: [(lat, lon, name), ...]}.
    """
    center = [stops_df["lat"].mean(), stops_df["lon"].mean()]

    m = folium.Map(
        location=center,
        zoom_start=11,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    trunk_layer  = folium.FeatureGroup(name="🚀 Trunk Routes",  show=True)
    feeder_layer = folium.FeatureGroup(name="🚌 Feeder Routes", show=True)
    stop_layer   = folium.FeatureGroup(name="Stops (all)",       show=False)

    bounds = []

    for rid, stop_list in routes.items():
        if not stop_list:
            continue

        # extract [lat, lon] regardless of tuple length
        coords = [[s[0], s[1]] for s in stop_list]
        bounds.extend(coords)

        if str(rid).startswith("TRUNK"):
            style = dict(color="#00E5FF", weight=6, opacity=0.95)
            layer = trunk_layer
        else:
            style = dict(color="#FF6B6B", weight=3, opacity=0.70)
            layer = feeder_layer

        folium.PolyLine(coords, **style, tooltip=str(rid)).add_to(layer)

    trunk_layer.add_to(m)
    feeder_layer.add_to(m)

    # Stops — small, semi-transparent, hidden by default
    for _, r in stops_df.iterrows():
        folium.CircleMarker(
            [r["lat"], r["lon"]],
            radius=2,
            color="#9AA4B2",
            fill=True,
            fill_opacity=0.35,
            weight=0,
        ).add_to(stop_layer)
    stop_layer.add_to(m)

    # Transfer hubs with glow
    if hubs_df is not None and len(hubs_df) > 0:
        for _, h in hubs_df.iterrows():
            # outer glow ring
            folium.CircleMarker(
                [h["lat"], h["lon"]],
                radius=16, color="#FFD54F",
                fill=True, fill_opacity=0.15, weight=0,
            ).add_to(m)
            # core marker
            folium.CircleMarker(
                [h["lat"], h["lon"]],
                radius=7,
                color="#FFC107", fill=True,
                fill_color="#FFC107", fill_opacity=0.95,
                popup=folium.Popup(
                    f"<b>{h['hub_id']}</b><br>🚌 {h['n_trunk_points']} trunk stops",
                    max_width=200,
                ),
                tooltip=h["hub_id"],
            ).add_to(m)

    # City boundary
    if boundary_gdf is not None:
        try:
            import json as _json
            folium.GeoJson(
                _json.loads(boundary_gdf.to_json()),
                name="City Boundary",
                style_function=lambda _: {
                    "fill": False, "color": "#00E5FF",
                    "weight": 2, "opacity": 0.6,
                },
            ).add_to(m)
        except Exception:
            pass

    if bounds:
        m.fit_bounds(bounds)

    folium.LayerControl(collapsed=False).add_to(m)
    return m



st.sidebar.header("⚙️ Control Panel")

mode = st.sidebar.radio(
    "Input Mode",
    ["CSV Coordinates", "Map Image (AI)"]
)

# New City Input
target_city = st.sidebar.text_input("📍 Target City (for Map Alignment)", "New York, NY")

uploaded_file = st.sidebar.file_uploader(
    "Upload Data",
    type=["csv", "png", "jpg", "jpeg"]
)

num_buses = st.sidebar.slider("🚌 Total Buses", 1, 100, 10)
operating_hours = st.sidebar.slider("⏱ Operating Hours", 1, 24, 12)

st.sidebar.markdown("---")
min_spacing_m = st.sidebar.slider("🚏 Min Stop Spacing (m)", 200, 800, 400, step=50)
bus_capacity = st.sidebar.slider("🚌 Bus Capacity (seats)", 30, 120, 60)
target_load_factor = st.sidebar.slider("🎯 Target Load Factor", 0.4, 1.0, 0.75, step=0.05)

st.sidebar.markdown("### ⏰ Temporal Service Tuning")
peak_multiplier = st.sidebar.slider("🚀 Peak Service Boost", 1.0, 3.0, 1.6, step=0.1)
offpeak_multiplier = st.sidebar.slider("🌙 Off-Peak Service Factor", 0.3, 1.0, 0.6, step=0.05)
peak_share = st.sidebar.slider("📊 Peak Hour Share", 0.2, 0.6, 0.35, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Smart Auto-Tune")
auto_mode = st.sidebar.toggle("🤖 Auto-Tune by City Scale", value=True)
strict_gtfs = st.sidebar.checkbox("🧪 Strict GTFS Validation", value=False)

run_btn = st.sidebar.button("🚀 Generate Smart Plan", use_container_width=True)

if "generated" not in st.session_state:
    st.session_state.generated = False

if run_btn:
    st.session_state.generated = True

# ================= HELPER =================
def show_progress():
    steps = [
        "Initializing AI pipeline...",
        "Running demand ML...",
        "Optimizing routes...",
        "Rendering maps..."
    ]
    progress = st.progress(0, text=steps[0])
    for i in range(100):
        if i == 10:
            progress.progress(i + 1, text=steps[1])
        elif i == 40:
            progress.progress(i + 1, text=steps[2])
        elif i == 70:
            progress.progress(i + 1, text=steps[3])
        else:
            progress.progress(i + 1)
        time.sleep(0.005)
    progress.empty()

def create_preview_map(df, routes=None):
    center = [df["lat"].mean(), df["lon"].mean()]
    try:
        m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    except:
        m = folium.Map(location=center, zoom_start=12)

    if routes:
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
                  'cadetblue', 'darkpurple', 'pink', 'lightblue', 'darkgreen', 'gray']

        # Download road graph once (cached)
        try:
            G = get_osm_graph(center[0], center[1])
            road_graph_ok = True
        except Exception:
            road_graph_ok = False
            G = None

        for i, (rid, route) in enumerate(routes.items()):
            color = colors[i % len(colors)]

            # Draw stop markers
            for lat, lon, name in route:
                folium.CircleMarker(
                    [lat, lon],
                    radius=4,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    popup=name
                ).add_to(m)

            # Road-following polyline
            if road_graph_ok:
                road_pts = get_road_path(G, route)
            else:
                road_pts = []

            if road_pts:
                folium.PolyLine(
                    road_pts,
                    color=color,
                    weight=4,
                    opacity=0.9
                ).add_to(m)
            else:
                # fallback: straight dashed line
                pts = [[lat, lon] for lat, lon, _ in route]
                folium.PolyLine(
                    pts,
                    color=color,
                    weight=3,
                    opacity=0.6,
                    dash_array="5,5"
                ).add_to(m)
    else:
        for _, row in df.iterrows():
            folium.CircleMarker(
                [row["lat"], row["lon"]],
                radius=4
            ).add_to(m)
    return m

def file_download_button(path):
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    fname = Path(path).name
    href = f'<a href="data:text/html;base64,{b64}" download="{fname}">⬇️ Download {fname}</a>'
    st.markdown(href, unsafe_allow_html=True)

# ================= MAIN LOGIC =================
if uploaded_file and st.session_state.generated:

    with st.spinner("🧠 AI is planning the bus network..."):
        show_progress()

        try:
            # -------- GEOCODING --------
            # -------- GEOCODING --------
            # -------- GEOCODING --------
            # Default coordinates (New York City) if geocoding fails
            base_lat, base_lon = 40.7128, -74.0060
            
            # Offline Fallback Dictionary for common cities (Instant & Works Offline)
            OFFLINE_CITIES = {
                "new york": (40.7128, -74.0060),
                "london": (51.5074, -0.1278),
                "paris": (48.8566, 2.3522),
                "tokyo": (35.6762, 139.6503),
                "mumbai": (19.0760, 72.8777),
                "delhi": (28.6139, 77.2090),
                "patan": (23.8512, 72.1266),
                "ahmedabad": (23.0225, 72.5714),
                "bangalore": (12.9716, 77.5946),
                "san francisco": (37.7749, -122.4194)
            }

            if mode == "Map Image (AI)":
                city_key = target_city.lower().split(",")[0].strip()
                
                # 1. Try Offline Dictionary First 
                if city_key in OFFLINE_CITIES:
                    base_lat, base_lon = OFFLINE_CITIES[city_key]
                    st.success(f"📍 Map aligned to: {target_city} (Offline Cache)")
                
                else:
                    # 2. Try Nominatim (OSM)
                    try:
                        geolocator = Nominatim(user_agent="BusAI_Planner_Universal", timeout=5)
                        location = geolocator.geocode(target_city)
                        if location:
                            base_lat = location.latitude
                            base_lon = location.longitude
                            st.success(f"📍 Map aligned to: {location.address}")
                        else:
                            raise Exception("City not found in OSM")
                            
                    except Exception:
                        # 3. Try ArcGIS (Fallback - often works when OSM is blocked)
                        try:
                            geolocator = ArcGIS(user_agent="BusAI_Planner_Universal", timeout=5)
                            location = geolocator.geocode(target_city)
                            if location:
                                base_lat, base_lon = location.latitude, location.longitude
                                st.success(f"📍 Map aligned to: {location.address} (via ArcGIS)")
                            else:
                                st.warning(f"⚠️ Could not find city '{target_city}'. Using default coordinates (NYC).")
                        except Exception:
                            st.warning(f"⚠️ Network Warning: Could not reach map services (OSM or ArcGIS). Using default coordinates. \nTry standard cities like 'New York', 'London', 'Mumbai' for offline support.")

            # -------- LOAD DATA --------
            if mode == "CSV Coordinates":
                uploaded_file.seek(0)
                try:
                    df = load_stops(uploaded_file)
                except UnicodeDecodeError:
                    st.error("❌ Error loading CSV: It looks like you uploaded a binary file (Image) but selected 'CSV Coordinates' mode. Please switch to 'Map Image (AI)' mode.")
                    st.stop()
            elif mode == "Map Image (AI)":
                uploaded_file.seek(0)
                df = detect_stops_from_image(uploaded_file, base_lat, base_lon)

            # -------- PIPELINE (CACHED) --------
            with st.spinner("🧠 Running multi-model transit AI..."):
                # Fetch city boundary (cached)
                boundary_gdf = cached_city_boundary(target_city, None, None)

                # Compute city scale + auto-tune parameters
                city_metrics = compute_city_metrics(boundary_gdf)
                if city_metrics and auto_mode:
                    city_scale = classify_city_scale(city_metrics["area_km2"])
                    auto_params = get_auto_parameters(city_scale)
                    eff_spacing    = auto_params["stop_spacing"]
                    eff_load       = auto_params["target_load"]
                    eff_peak_boost = auto_params["peak_boost"]
                    eff_off_factor = auto_params["offpeak_factor"]
                    n_cl_override  = max(4, int((city_metrics["area_km2"] / 25) * auto_params["cluster_factor"]))
                    st.sidebar.info(f"🤖 Auto-tuned: **{city_scale.replace('_', ' ').title()}** ({city_metrics['area_km2']:.0f} km²)")
                else:
                    city_scale    = "unknown"
                    auto_params   = {}
                    eff_spacing   = min_spacing_m
                    eff_load      = target_load_factor
                    eff_peak_boost= peak_multiplier
                    eff_off_factor= offpeak_multiplier
                    n_cl_override = None

                # Clip stops to city boundary
                if boundary_gdf is not None:
                    df = clip_points_to_boundary(df, boundary_gdf)
                    if len(df) == 0:
                        st.error("❌ No stops found inside detected city boundary. Try a different city name.")
                        st.stop()

                df_tuple = (tuple(df.columns), [tuple(r) for r in df.values])
                demand_df, rmse, importance_df, df, routes, allocation, metrics, load_df, events_df, corridors_df, hubs_df = cached_pipeline(
                    df_tuple, num_buses, operating_hours,
                    min_spacing_m=eff_spacing,
                    bus_capacity=bus_capacity,
                    target_load_factor=eff_load,
                    peak_multiplier=eff_peak_boost,
                    offpeak_multiplier=eff_off_factor,
                    peak_share=peak_share,
                    n_clusters_override=n_cl_override
                )

            st.success(f"🧠 AI Demand Model RMSE: {rmse:.2f}")
            st.caption("Models: XGBoost + CNN + GNN Ensemble")

            # --- AI PRESENTATION MAP (OSMNX, CACHED) ---
            try:
                with st.spinner("🎨 Creating presentation-quality map with OSMnx..."):
                    beautiful_map_path = cached_beautiful_map(
                        routes, num_buses, base_lat, base_lon
                    )

                    if beautiful_map_path:
                        st.success("✨ AI Map Generated Successfully!")
                        st.image(beautiful_map_path, caption="AI-Generated Transit Plan Visualization", width="stretch")
                    else:
                        st.info("ℹ️ OSMnx Map could not be generated.")
            except Exception as e:
                st.warning(f"⚠️ OSMnx Map Generation skipped: {e}")

            paths = create_bus_maps(df, routes)
            
            # ================= TABS =================
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Dashboard",
                "🔥 Demand Heatmap",
                "🗺 Routes Preview",
                "📈 Load Simulation",
                "🛣 Corridor Analysis",
                "⬇️ Downloads",
            ])

            # ================= DASHBOARD =================
            with tab1:
                render_kpi_row(metrics, rmse, city_scale, hubs_df, routes, events_df)
                st.divider()

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.metric("Total Stops", metrics["total_stops"])
                with c2:
                    st.metric("Routes", metrics["total_routes"])
                with c3:
                    st.metric("Buses Used", metrics["buses_used"])
                with c4:
                    st.metric("Coverage", f'{metrics["coverage_score"]}%')

                c5, c6, c7 = st.columns(3)
                with c5:
                    st.metric("Avg Headway", f'{metrics.get("avg_headway_hours", "—")} hrs')
                with c6:
                    st.metric("Model RMSE", f"{rmse:.2f}")
                with c7:
                    st.metric("🚨 Adaptive Interventions", len(events_df))

                # Row 3: city + network intelligence KPIs
                c8, c9, c10 = st.columns(3)
                with c8:
                    st.metric("🏙 City Scale", city_scale.replace("_", " ").title())
                with c9:
                    trunk_count = sum(1 for r in routes if str(r).startswith("TRUNK"))
                    st.metric("🚌 Trunk Corridors", trunk_count)
                with c10:
                    feeder_count = sum(1 for r in routes if str(r).startswith("FEEDER"))
                    st.metric("🚻 Feeder Routes", feeder_count)

                st.markdown("### 🚌 Bus Allocation")
                alloc_display = pd.DataFrame(allocation).T
                st.dataframe(alloc_display, width="stretch")

                # Capacity utilization chart (show only if freq optimizer ran)
                if "capacity_utilization" in alloc_display.columns:
                    st.markdown("### 📈 Service Quality — Capacity Utilisation")
                    st.bar_chart(alloc_display["capacity_utilization"].astype(float))

                # Peak / off-peak scheduling table
                peak_cols = [c for c in ["peak_buses", "offpeak_buses",
                                          "peak_headway_min", "offpeak_headway_min",
                                          "service_intensity"] if c in alloc_display.columns]
                if peak_cols:
                    st.markdown("### ⏰ Peak vs Off-Peak Service")
                    st.dataframe(alloc_display[peak_cols], width="stretch", hide_index=False)

                # Transfer hub summary
                if len(hubs_df) > 0:
                    st.markdown("### 🔁 Transfer Hubs")
                    st.dataframe(hubs_df, width="stretch", hide_index=True)

            # ================= DEMAND HEATMAP =================
            with tab2:
                st.markdown("### 🔥 AI Predicted Transit Demand")
                st.caption(f"Ensemble Model Prediction (XGBoost + CNN + GNN · RMSE: {rmse:.2f})")

                fig = px.density_map(
                    demand_df,
                    lat="lat",
                    lon="lon",
                    z="final_demand",
                    radius=30,
                    center=dict(lat=demand_df["lat"].mean(), lon=demand_df["lon"].mean()),
                    zoom=10,
                    map_style="open-street-map",
                    color_continuous_scale="Turbo",
                )
                st.plotly_chart(fig, width="stretch")

                st.markdown("### 📊 Feature Importance")
                st.bar_chart(importance_df.set_index("feature"))

            # ================= ROUTE PREVIEW =================
            with tab3:
                st.markdown("### 🗺️ Routes & Transfer Hub Preview")

                preview_map = render_routes_and_hubs(
                    routes, df, hubs_df, boundary_gdf
                )
                st_folium(preview_map, height=620, width="stretch")

                st.markdown(
                    "<div style='color:#9AA4B2;font-size:13px;margin-top:6px'>"
                    "🚀 <b style='color:#00E5FF'>Cyan thick</b> = trunk routes &nbsp;·&nbsp; "
                    "🚌 <b style='color:#FF6B6B'>Red thin</b> = feeder routes &nbsp;·&nbsp; "
                    "🟡 <b style='color:#FFC107'>Gold glow</b> = transfer hubs &nbsp;·&nbsp; "
                    "<span style='color:#00E5FF'>Dashed</span> = city boundary"
                    "</div>",
                    unsafe_allow_html=True,
                )

            # ================= LOAD SIMULATION =================
            with tab4:
                st.markdown("### 📈 Hourly Passenger Load & Adaptive Response")
                st.caption("Stochastic Poisson model · peak/off-peak aware · adaptive dispatch included")

                route_ids = sorted(load_df["route_id"].unique())
                route_choice = st.selectbox("Select Route", route_ids)
                plot_df = load_df[load_df["route_id"] == route_choice]

                # Plotly chart with capacity limit line
                fig_load = go.Figure()
                fig_load.add_trace(go.Scatter(
                    x=plot_df["hour"],
                    y=plot_df["load_factor"],
                    mode="lines+markers",
                    name="Load Factor",
                    line=dict(color="#6c63ff", width=2.5),
                    marker=dict(size=6),
                ))
                fig_load.add_trace(go.Bar(
                    x=plot_df["hour"],
                    y=plot_df["passengers"],
                    name="Passengers",
                    marker_color="rgba(255,107,107,0.35)",
                    yaxis="y2",
                ))
                fig_load.add_hline(
                    y=1.0,
                    line_dash="dash",
                    line_color="#ff4b4b",
                    annotation_text="Capacity Limit",
                    annotation_position="top left",
                )
                fig_load.update_layout(
                    height=400,
                    xaxis_title="Hour of Day",
                    yaxis_title="Load Factor",
                    yaxis2=dict(title="Passengers", overlaying="y", side="right"),
                    legend=dict(orientation="h", y=1.1),
                    margin=dict(t=20, b=20),
                )
                st.plotly_chart(fig_load, width="stretch")

                # Adaptive intervention summary for this route
                route_events = events_df[events_df["route_id"] == route_choice] if len(events_df) > 0 else pd.DataFrame()

                if len(route_events) > 0:
                    st.warning(f"🚨 {len(route_events)} adaptive intervention(s) triggered for this route")
                    st.dataframe(route_events, use_container_width=True)
                    unresolved = route_events[route_events["resolved"] == False]
                    if len(unresolved):
                        st.error(f"❌ {len(unresolved)} hour(s) remain overcrowded after intervention")
                    else:
                        st.success("✅ All overcrowding resolved after extra-bus dispatch")
                else:
                    st.success("✅ No adaptive intervention required for this route")

                st.markdown("#### 📋 Hourly Detail")
                detail_cols = [c for c in [
                    "hour", "passengers", "capacity_per_hour",
                    "load_factor", "overcrowded", "intervention_applied"
                ] if c in plot_df.columns]
                st.dataframe(plot_df[detail_cols], width="stretch")

            # ================= CORRIDOR ANALYSIS =================
            with tab5:
                st.markdown("### 🛣 Detected High-Demand Corridors")
                st.caption("DBSCAN spatial clustering · PCA linearity check · BRT candidate scoring")

                if corridors_df is None or len(corridors_df) == 0:
                    st.info("ℹ️ No strong linear corridors detected. Try uploading denser stop data.")
                else:
                    brt_count = int(corridors_df["is_brt_candidate"].sum())
                    ca, cb, cc = st.columns(3)
                    with ca: st.metric("Corridors Found", len(corridors_df))
                    with cb: st.metric("🚌 BRT Candidates", brt_count)
                    with cc: st.metric("📏 Avg Length (km)", round(corridors_df["length_km"].mean(), 1))

                    # Demand score bar chart
                    fig_corr = px.bar(
                        corridors_df,
                        x="corridor_id",
                        y="demand_score",
                        color="is_brt_candidate",
                        color_discrete_map={True: "#00E5FF", False: "#6c63ff"},
                        labels={"demand_score": "Demand Score", "corridor_id": "Corridor ID",
                                "is_brt_candidate": "BRT Candidate"},
                        title="📊 Corridor Demand Scores",
                        height=350,
                    )
                    fig_corr.update_layout(margin=dict(t=40, b=10))
                    st.plotly_chart(fig_corr, width="stretch")

                    st.dataframe(corridors_df, width="stretch", hide_index=True)

                    if brt_count > 0:
                        st.success(
                            f"✅ {brt_count} corridor(s) meet BRT criteria (≥6 km, demand >500). "
                            "Consider dedicated lanes on these spines."
                        )

            # ================= DOWNLOADS =================
            with tab6:
                st.markdown("### 📁 Download Individual Bus Routes")
                for p in paths:
                    file_download_button(p)
                    st.divider()

                st.markdown("### 🧧 GTFS Feed (Transit Standard)")
                try:
                    gtfs_path = export_gtfs(df, routes, allocation)
                    gtfs_warnings = validate_and_fix_gtfs("outputs/gtfs")

                    if gtfs_warnings:
                        if strict_gtfs:
                            st.error("❌ Strict GTFS validation failed:")
                        else:
                            st.warning("⚠️ GTFS Auto-Fixes Applied:")
                        for w in gtfs_warnings:
                            st.write(f"- {w}")
                    else:
                        st.success("✅ GTFS validated successfully (no issues found)")

                    file_download_button(gtfs_path)
                except Exception as e:
                    st.warning(f"⚠️ GTFS export skipped: {e}")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())

else:
    st.info("👈 Upload data and click **Generate Smart Plan** to begin.")
