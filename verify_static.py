import urllib.request
try:
    print("Checking http://localhost:8081/static/login.html...")
    resp = urllib.request.urlopen("http://localhost:8081/static/login.html")
    print(f"Status: {resp.getcode()}")
    print("Content-Type:", resp.headers.get_content_type())
except Exception as e:
    print(f"Failed: {e}")
