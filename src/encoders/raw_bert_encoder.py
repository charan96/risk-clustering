from bert_serving.client import BertClient

from encoders.encoder import Encoder


class RawBERTEncoder(Encoder):
    def __init__(self):
        self.client = BertClient(check_length=False)
        
    def encode(self, data):
        return self.client.encode([data]).tolist()[0]

    def encode_multiple(self, data):
        return self.client.encode(data).tolist()
