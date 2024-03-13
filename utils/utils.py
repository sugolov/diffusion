
from datetime import datetime

def timestamp(label):

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    label = str(label) if not isinstance(label, str) else label
    print(str(current_time) + ": " + label)