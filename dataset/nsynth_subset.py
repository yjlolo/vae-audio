"""
This script should be placed in the same directory as the folder 'full'.
It is to make a subset from the full nsynth dataset in order to follow GANSYNTH
https://openreview.net/forum?id=H1xQVn09FX.

/path/to/nsynth/
    -full/
        -nsynth-train/
        -nsynth-valid/
        -nsynth-test/
    -nsynth_subset.py

The outcome is a folder named 'subset' containing only acoustic notes within pitch MIDI 24-84.
"""

import shutil
from pathlib import Path


path_to_nsynth = Path('/data/yinjyun/datasets/nsynth')
path_to_full = path_to_nsynth / 'full'
path_to_subset = path_to_nsynth / 'subset' / 'audio'

if not path_to_subset.is_dir():
    path_to_subset.mkdir(parents=True, exist_ok=False)

full_train = path_to_full / 'nsynth-train' / 'audio'
full_valid = path_to_full / 'nsynth-valid' / 'audio'
full_test = path_to_full / 'nsynth-test' / 'audio'

assert full_train.exists()
assert full_valid.exists()
assert full_test.exists()

def filter_file(f):
    """
    Filter the audio files based on the criteria described in GANSYNTH:
    1. acoustics
    2. fall in pitch MIDI [24, 84]
    This however results in 86775 samples in total (whereas there are only 70379 in the paper).
    But the full dataset size in paper (300000) is also different from the downloaded one (305979).
    """
    f.stem.split('-')
    instrument, pitch, vel = f.stem.split('-')[0], int(f.stem.split('-')[1]), int(f.stem.split('-')[2])
    source = instrument.split('_')[-2]
    # print(f.stem, source, pitch, vel)
    assert source in ['acoustic', 'electronic', 'synthetic']
    assert pitch >= 0 and pitch <= 127
    assert vel in [25, 50, 75, 100, 127]

    include_in_subset = True
    if (source != 'acoustic') or (pitch < 24) or (pitch > 84):
        include_in_subset = False
    return include_in_subset


print("The size of original dataset: %d"
      % (len(list(full_train.glob('*.wav'))) + len(list(full_valid.glob('*.wav'))) + len(list(full_test.glob('*.wav')))))
filter_train = [i for i in full_train.glob('*.wav') if filter_file(i)]
filter_valid = [i for i in full_valid.glob('*.wav') if filter_file(i)]
filter_test = [i for i in full_test.glob('*.wav') if filter_file(i)]
filter_set = filter_train + filter_valid + filter_test
print("The size of filtered dataset: %d" % len(filter_set))

print("copy and the filtered data to %s ..." % str(path_to_subset))
for i in filter_set:
    shutil.copy(str(i), path_to_subset)
