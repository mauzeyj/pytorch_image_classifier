import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image


def transformations():
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),

                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    return {'train_transformations': train_transforms,
            'test_validation_transformations': test_transforms}


def loaders(data_dir, train_dir, valid_dir, test_dir, transforms):
    train_data = datasets.ImageFolder(train_dir, transform=transforms['train_transformations'])
    valid_data = datasets.ImageFolder(valid_dir, transform=transforms['test_validation_transformations'])
    test_data = datasets.ImageFolder(test_dir, transform=transforms['test_validation_transformations'])

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=32,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=32,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=32,
                                              shuffle=True)
    return {'train_loader': train_loader,
            'valid_loader': valid_loader,
            'test_loader': valid_loader}


def standard_folders_config():
    standard = {'data_dir' : 'flowers',
            'train_dir' : 'flowers/train',
            'valid_dir' : 'flowers/valid',
            'test_dir' :  'flowers/test'}
    return standard

def get_labels():
    with open('cat_to_name.json', 'r') as f:
        labels = json.load(f)
    return labels


def combine_model_classifer():

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def process_image(image, transformations):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    #     np_image = np.array(im).T
    loader = transformations['test_validation_transformations']
    im = loader(im).float().numpy()
    #     im = torch.autograd.variable(im, requires_grad=True)
    #     im = im.unsqueeze(0)
    return im


def validation(model, loader, criterion):
    test_loss = 0
    accuracy = 0
    device = get_device()
    for images, labels in loader:
        #         images.resize_(images.shape[0], 784)
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy