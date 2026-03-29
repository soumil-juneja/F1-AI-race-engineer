import requests
import time

def test_openf1_connection():
    # Use the 'sessions' endpoint as a baseline health check
    test_url = "https://api.openf1.org/v1/sessions?session_key=9472"
    
    print(f"--- TESTING OPENF1 CONNECTION ---")
    try:
        start_time = time.time()
        response = requests.get(test_url, timeout=5)
        latency = (time.time() - start_time) * 1000
        
        # Check 1: HTTP Status
        if response.status_code == 200:
            print(f"✅ STATUS: 200 OK")
        else:
            print(f"❌ STATUS: {response.status_code} ({response.reason})")
            return

        # Check 2: Latency
        print(f"⏱️ LATENCY: {latency:.2f}ms")

        # Check 3: Data Integrity (Is it valid JSON and a list?)
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            print(f"📦 DATA: Received {len(data)} session record(s). Integrity check PASSED.")
        else:
            print(f"⚠️ DATA: Response is empty or malformed.")

    except requests.exceptions.Timeout:
        print("❌ ERROR: Connection timed out. The API might be down or your internet is slow.")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_openf1_connection()