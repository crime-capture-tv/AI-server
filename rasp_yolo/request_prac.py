import requests, json
store_no = 1
# URL = f"http://192.168.0.49:8080/api/v1/stores/{store_no}/videos"
URL = "http://192.168.0.49:8080/api/v1/stores/1/videos?storeNo=1"

# data with json
data = {
    "suspicionVideoPath": "\\\\192.168.0.42\\crimecapturetv\\suspicion-video\\20230916_152156539.mp4"
}

headers = {
    "accept": "*/*",
    "Content-Type": "application/json"
}

response = requests.post(URL, json=data, headers=headers)
print(response.status_code)
print(response.request.body.decode())
