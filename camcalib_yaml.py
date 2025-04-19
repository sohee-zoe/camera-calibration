import yaml
import numpy as np

def save_calibration_to_yaml(filename, K, D):
    data = {
        "K": K.tolist(),
        "D": D.tolist()
    }
    with open(filename, 'w') as f:
        yaml.dump(data, f)

def load_calibration_from_yaml(filename):
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    K = np.array(data["K"])
    D = np.array(data["D"])
    return K, D