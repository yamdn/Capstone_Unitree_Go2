import requests

url = "http://localhost:5051/action/heart"   # Update with your actual server address

payload = {
#   "vx": 1.0,   # between -2.5 and 3.8
#   "vy": 0.0,   # between -1.0 and 1.0
#   "yaw": 0.5   # between -4.0 and 4.0
}

response = requests.post(url, json=payload)

print("Status code:", response.status_code)
print("Response:", response.json())
