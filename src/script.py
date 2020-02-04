import os
import glob

import config as cfg
from encoders.tfidf_encoder import TFIDFEncoder


encoder = TFIDFEncoder()

files = glob.glob(os.path.join(cfg.DATA_DIR, '*/*.risk'))

data = []

for f in files:
    with open(f, 'r') as fh:
        data.append(*fh.readlines())

encoder.encode_multiple(data)
