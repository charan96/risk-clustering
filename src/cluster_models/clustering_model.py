from abc import ABC, abstractmethod


class ClusteringModel(ABC):
    @abstractmethod
    def cluster(self, x, n_clusters):
        pass

    @abstractmethod
    def get_stockname_label_pairs(self, stocknames):
        pass

    def print_stockname_label_pairs(self, stockname_label_pairs):
        print('STOCKNAME:', 'LABEL')
        print('----------------')
        for stockname, label in stockname_label_pairs:
            print('{:4s}: {:2d}'.format(stockname, label))
