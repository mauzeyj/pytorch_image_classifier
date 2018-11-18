import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import OrderedDict
from functions import transformations, get_device, pro
from PIL import Image

def process_image(image, transformations):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)

    loader = transformations['test_validation_transformations']
    im = loader(im).float().numpy()

    return im

def get_labels():
    with open('cat_to_name.json', 'r') as f:
        labels = json.load(f)
        labels = {int(k):v for k,v in labels.items()}
    return labels

def make_prediction(image_path, model):
    model.eval()
    return model.forward(torch.from_numpy(process_image(image_path, transformations())).unsqueeze(0).to(get_device()))

def predict(image_path, model, topk = 5):
    model.eval()
    with torch.no_grad():
        prediction = model.forward(torch.from_numpy(process_image(image_path, transformations())).unsqueeze(0).to(get_device())).topk(topk)
    names = get_labels()
    keys = []
    for n in prediction[1].numpy()[0]:
        keys.append(names[n])
    return prediction[0].numpy()[0], prediction[1].numpy()[0], np.array(keys)
