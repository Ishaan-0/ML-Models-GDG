import requests
import json

url = 'http://172.20.10.5:8080/'
data = {
        "N": 50,
        "P": 30,
        "K": 20,
        "humidity": 60.5,
        "rainfall": 150.2,
        "ph": 6.8
    }
headers = {'Content-Type': 'application/json'}

response = requests.post(url, data=json.dumps(data), headers=headers)

print(response.json())