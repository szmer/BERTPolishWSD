import re

def is_nonalphabetic(text):
    return len(text) > 0 and len(re.sub('[^\\W0-9]', '', text)) / len(text) >= 0.65

def sequence_differences(seq1, seq2):
    """
    Return a list of pairs of differing items in seq1 and seq2 (they have to be of the same length).
    """
    assert len(seq1) == len(seq2)
    diffs = []
    for item_n, item in enumerate(seq1):
        if item != seq2[item_n]:
            diffs.append((item, seq2[item_n]))
    return diffs
