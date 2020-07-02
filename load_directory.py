import os

INPUT_URL = "/home/susi/darknet/data/images"
data = os.listdir(INPUT_URL)
for d in data:
    print(os.path.join(INPUT_URL, d))