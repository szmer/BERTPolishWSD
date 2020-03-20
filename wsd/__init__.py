import re

def is_nonalphabetic(text):
    return len(text) > 0 and len(re.sub('[^\\W0-9]', '', text)) / len(text) >= 0.65
