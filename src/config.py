from encoders.tfidf_encoder import TFIDFEncoder
from encoders.raw_bert_encoder import RawBERTEncoder
from cluster_models.k_means_clustering_model import KMeansClusteringModel

# location where the raw HTM files of 10-K filings are located
DATA_DIR = '../data'

# location of the risk files directory
RISK_FILES_DIR = '../data/risk_files'

# location where the encoded data should be written to
ENCODED_DATA_DIR = '../data/encoded_data'

# uses precomputed risk files located in RISK_FILES_DIR
USE_PRECOMPUTED_RISK_FILES = True

# uses precomputed encoded vectors located in ENCODED_DATA_DIR
USE_PREENCODED_VECTORS = False

# K-Means random seed
RANDOM_SEED = 100

# N-grams for TFIDF
N_GRAMS = 4

# encoder to be used for obtaining the encoded representation of the text
ENCODER = TFIDFEncoder()

# the clustering model to be used
CLUSTER_MODEL = KMeansClusteringModel(ENCODED_DATA_DIR, RANDOM_SEED)

