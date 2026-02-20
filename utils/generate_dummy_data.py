import json
import random

def generate_patan_data(filename="patan_stops.geojson", num_stops=50):
    # Patan, Gujarat coordinates
    center_lat = 23.8512
    center_lon = 72.1266
    
    features = []
    
    for i in range(num_stops):
        # Generate random offset
        lat_offset = random.uniform(-0.03, 0.03)
        lon_offset = random.uniform(-0.03, 0.03)
        
        stop_lat = center_lat + lat_offset
        stop_lon = center_lon + lon_offset
        
        feature = {
            "type": "Feature",
            "properties": {
                "name": f"Patan Stop {i+1}",
                "id": i
            },
            "geometry": {
                "type": "Point",
                "coordinates": [stop_lon, stop_lat]  # GeoJSON is [lon, lat]
            }
        }
        features.append(feature)
    
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(filename, "w") as f:
        json.dump(geojson_data, f, indent=2)
    
    print(f"Generated {filename} with {num_stops} stops.")

if __name__ == "__main__":
    generate_patan_data()
