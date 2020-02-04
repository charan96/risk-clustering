# How To Run

1. Start the BERT Server using the following command repo root: `bert-serving-start -model_dir bert/uncased_L-24_H-1024_A-16/`

2. The entrypoint script can be run from `src/` using: `python main.py`


# NOTE
1. `src/config.py` contains all the configurable values.

2. BERT Server needs to be running for any BERT-based encoders to work.
