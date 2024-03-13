import json
import os
from datetime import datetime

def timestamp(label):

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    label = str(label) if not isinstance(label, str) else label
    print(str(current_time) + ": " + label)

def save_config(config, name, location=""):
    file_name = location + name + ".json"
    print(os.getcwd())
    with open(file_name, 'w') as file:
        json.dump(config, file)