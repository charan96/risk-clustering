import os
import glob
import json
import numpy as np
from pprint import pprint
from scipy.spatial.distance import cosine as cos
from scipy.spatial.distance import euclidean as ec

import config as cfg
from risk_file_generator import RiskFileGenerator


raw_files = glob.glob(os.path.join(cfg.DATA_DIR, '*/*.htm*'))

rfg = RiskFileGenerator(cfg.DATA_DIR)

# ENCODER AND CLUSTERING MODELS
encoder = cfg.ENCODER
cluster_model = cfg.CLUSTER_MODEL

stocknames = []
risk_filenames = []
encoded_filenames = []

if not cfg.USE_PRECOMPUTED_RISK_FILES:
    for raw_file in raw_files:
        print('PROCESSING RAW FILE: {}'.format(raw_file))
        stock_name = raw_file.split('/')[-2]
        stocknames.append(stock_name)

        risk_filename = rfg.get_risk_filename_from_raw_file(raw_file)
        risk_filenames.append(risk_filename)

        rfg.generate_risk_file(raw_file)
else:
    risk_filenames = glob.glob(os.path.join(cfg.RISK_FILES_DIR, '*.risk'))

if not cfg.USE_PREENCODED_VECTORS:
    risk_file_data = []
    
    for risk_filename in risk_filenames:
        stock_name = risk_filename.split('/')[-1].split('.')[0]
        encoded_filename = encoder.get_encoded_filename(cfg.ENCODED_DATA_DIR, stock_name + '.json')

        risk_file_data.append(encoder.load_risk_file(risk_filename))
        encoded_filenames.append(encoded_filename)

    vectors = encoder.encode_multiple(risk_file_data)
    stocknames = []

    for vector, encoded_filename in zip(vectors, encoded_filenames):
        stocknames.append(encoded_filename)
        print('SAVING ENCODED VECTOR AT: {}'.format(encoded_filename))
        encoder.save_encoded_vector(vector, encoded_filename)
else:
    encoded_filenames = glob.glob(os.path.join(cfg.ENCODED_DATA_DIR, '*.json'))
    stocknames = [sn.split('/')[-1].split('.')[0] for sn in encoded_filenames]
        
vectors, vector_stock_map = cluster_model.load_encoded_vectors_files(encoded_filenames)
kmeans_model = cluster_model.cluster(vectors, n_clusters=4)
stockname_label_pairs = cluster_model.get_stockname_label_pairs(stocknames)

print()
cluster_model.print_stockname_label_pairs(stockname_label_pairs)

# use PCA to remove dimensions common to all of the stocks; thereby ideally removing common risk factors to all the stocks

