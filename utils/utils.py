import json
import os
from datetime import datetime

def timestamp(label):

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    label = str(label) if not isinstance(label, str) else label
    print(str(current_time) + ": " + label)

def save_dict_to_json(dictionary, name, location=""):
    file_name = location + name + ".json"
    with open(file_name, 'w') as file:
        json.dump(dictionary, file)
    file.close()

def load_dict_from_json(name, location=""):
    file = open(
        os.path.join(location, name + ".json")
    )
    config = json.load(file)
    file.close()
    return config