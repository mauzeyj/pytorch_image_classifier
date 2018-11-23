import argparse
import json

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


def predict(image_path, model, topk=5):
    model.eval()
    with torch.no_grad():
        prediction = model.forward(
            torch.from_numpy(process_image(image_path, transformations())).unsqueeze(0).to(get_device())).topk(topk)
    names = get_labels()
    keys = []
    for n in prediction[1].numpy()[0]:
        keys.append(names[n])
    return prediction[0].numpy()[0], prediction[1].numpy()[0], np.array(keys)


def load_classifier(path):
    check = torch.load(path, map_location=lambda storage, location: storage)
    classifier = Network(check['input_size'],
                         check['output_size'],
                         check['hidden_layers'],
                         check['drop_out'])
    classifier.load_state_dict(check['state_dict'])
    return classifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image-location', required=True)
    args = parser.parse_args()
    args = args.__dict__
    classifier, tl_model = load_classifier('checkpoint.pth')
    model = load_transfer_model(tl_model)
    model.classifier = classifier
    predict(args['image_location'])
