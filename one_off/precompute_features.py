import os 
import numpy as np
from config import STRING_MIDI
from audio_features import process_track, get_mirdata_dataset, get_solo_track_ids

CACHE_DIR = 'data/processed'

def precompute_all_solo_tracks():
    pass