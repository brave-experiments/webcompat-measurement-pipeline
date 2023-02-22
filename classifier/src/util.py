import os
import pickle
from src.config import *

def split_graph_id_and_label(graph_id_with_label):
    return tuple(graph_id_with_label.split('-'))


def save_as_pickle(obj, filename):
    with open(os.path.join(BASE_PATH, TRAINING_DIR, filename), 'wb') as f:
        return pickle.dump(obj, f)


def load_from_pickle(filename):
    with open(os.path.join(BASE_PATH, TRAINING_DIR, filename), 'rb') as f:
        return pickle.load(f)