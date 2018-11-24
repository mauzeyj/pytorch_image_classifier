import argparse
import json

import numpy as np
import torch
from PIL import Image

from functions import transformations, get_device
from models import load_transfer_model, Network


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
        labels = {int(k): v for k, v in labels.items()}
    return labels


def make_prediction(image_path, model):
    model.eval()
    return model.forward(torch.from_numpy(process_image(image_path, transformations())).unsqueeze(0).to(get_device()))


def predict(image_path, model, class_to_idx, topk=5):

    model.eval()
    img = process_image(image_path, transformations())
    for_model = torch.from_numpy(img).unsqueeze(0)
    probabilities = torch.exp(model.forward(for_model))
    top_probabilities, top_labels = probabilities.topk(topk)
    names = get_labels()
    idx_to_class = {val: key for key, val in class_to_idx.items()}
    keys = []
    for n in top_labels.detach().numpy()[0]:
        a = idx_to_class[n]
        keys.append(names[int(a)])
    return top_probabilities.detach().numpy()[0], top_labels.detach().numpy()[0], np.array(keys)


def load_classifier(path):
    check = torch.load(path, map_location=lambda storage, location: storage)
    classifier = Network(check['input_size'],
                         check['output_size'],
                         check['hidden_layers'],
                         check['drop_out'])
    classifier.load_state_dict(check['state_dict'])
    return classifier, check['transfer_learning_model'], check['class_to_idx']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-location', required=True)
    parser.add_argument('--topk', default=1, help='Number of top classes to show')
    args = parser.parse_args()
    args = args.__dict__
    classifier, tl_model, class_to_idx = load_classifier('checkpoint.pth')
    model = load_transfer_model(tl_model)
    model.classifier = classifier
    predictions, labels, names = predict(args['image_location'], model, class_to_idx, args['topk'])
    for x in range(len(predictions)):
        print('The image was given a predicttion value of {}, for label of {}'.format(predictions[x], names[x]))
