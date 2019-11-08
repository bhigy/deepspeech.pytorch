import json
import numpy as np
import os


"""Load alignment data."""
alignment_path = '/roaming/gchrupal/fanta/librispeech_val.fa.json'
# Reading output of forced-alignment
valid = []
for line in open(alignment_path):
    item = json.loads(line)
    # Filtering cases where alignement fail
    if np.all([word.get('start', False) for word in item['words']]):
        audio_id = os.path.basename(item['audiopath'])
        print(audio_id)
