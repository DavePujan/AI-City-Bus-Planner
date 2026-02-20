import requests
import socket

def check_connectivity():
    print("----------------------------------------------------------------")
    print("🔌 Network Connectivity Diagnostic")
    print("----------------------------------------------------------------")

    # 1. Check DNS resolution
    targets = [
        ("google.com", 80),
        ("nominatim.openstreetmap.org", 80),
        ("tile.openstreetmap.org", 80)
    ]

    for host, port in targets:
        try:
            ip = socket.gethostbyname(host)
            print(f"✅ DNS Resolution for {host}: {ip}")
        except socket.gaierror:
            print(f"❌ DNS Resolution Failed for {host}")
        except Exception as e:
            print(f"❌ DNS/Socket Error for {host}: {e}")

    print("\n----------------------------------------------------------------")

    # 2. Check HTTP Requests
    urls = [
        "https://nominatim.openstreetmap.org/search?q=New+York&format=json",
        "https://tile.openstreetmap.org/0/0/0.png"
    ]

    headers = {
        "User-Agent": "BusAI_Planner_Diagnostic/1.0"
    }

    for url in urls:
        print(f"Testing URL: {url}")
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                print(f"✅ Success! Status Code: {response.status_code}")
                print(f"   Content Size: {len(response.content)} bytes")
            else:
                print(f"⚠️ Failed. Status Code: {response.status_code}")
        except requests.exceptions.SSLError as e:
            print(f"❌ SSL Error: {e}")
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection Error: {e}")
        except Exception as e:
            print(f"❌ General Error: {e}")
        print("-" * 30)

if __name__ == "__main__":
    check_connectivity()
