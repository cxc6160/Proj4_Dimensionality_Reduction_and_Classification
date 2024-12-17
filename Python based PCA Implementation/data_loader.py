import os
import numpy as np
import matplotlib.pyplot as plt

def load_faces(data_dir):
    images = []
    labels = []
    for label, person in enumerate(sorted(os.listdir(data_dir))):
        person_dir = os.path.join(data_dir, person)
        if os.path.isdir(person_dir):
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = plt.imread(img_path).astype(np.float64)
                images.append(img.flatten())
                labels.append(label)
    return np.array(images), np.array(labels)
