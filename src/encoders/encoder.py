import os
import json
from abc import ABC, abstractmethod


class Encoder(ABC):
    def load_risk_file(self, risk_filename):
        with open(risk_filename, 'r') as fh:
            data = fh.readlines()
        
        return data[0]
   
    def save_encoded_vector(self, vector, filename):
        with open(filename, 'w') as fh:
            json.dump(vector, fh, indent=4)

    def get_encoded_filename(self, dir_loc, filename):
        if not os.path.exists(dir_loc):
            os.makedirs(dir_loc)
        
        return os.path.join(dir_loc, filename)

    @abstractmethod
    def encode_multiple(self, data):
        pass
