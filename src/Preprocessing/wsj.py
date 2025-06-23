import pickle

class WSJ:
    def __init__(self):
        return None

    def load(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)