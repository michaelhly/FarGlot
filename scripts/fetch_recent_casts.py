import os
import json
import requests

CAST_BATCHES = 1
OUT_PATH = ""
API_SECRET = os.getenv("MERKLE_SECRET")
HTTP_HEADERS = {"Authorization": f"Bearer {API_SECRET}"}

if not API_SECRET:
    raise Exception("MERKLE_SECRET not set")

with open("data/recent-casts.json", "w+") as f:
    all_casts = []
    cursor = None
    for i in range(CAST_BATCHES):
        url = "https://api.warpcast.com/v2/recent-casts?limit=1000"
        if cursor:
            url += f"&cursor={cursor}"
        res = requests.get(url=url, headers=HTTP_HEADERS)
        data = res.json()

        cursor = data["next"]["cursor"]
        all_casts.extend(data["result"]["casts"])

    if OUT_PATH:
        out = open(OUT_PATH, "w")
        out.write(json.dumps(all_casts))
    else:
        f.write(json.dumps(all_casts))
