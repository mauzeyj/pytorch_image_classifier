import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import OrderedDict
from functions import get_device, standard_folders_config, transformations, loaders, validation
from models import Network, load_transfer_model


def train(hidden_layers, epochs, drop_out = 0.5, core_model = 'vgg16'):
    folders = standard_folders_config()
    classifier = Network(25008, 102, hidden_layers, drop_out)
    model = load_transfer_model()
    model.classifier = classifier
    my_loaders = loaders(folders['data_dir'],
                         folders['train_dir'],
                         folders['valid_dir'],
                         folders['test_dir'],
                         transformations())
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    with active_session():
        device = get_device()
        train_loader = my_loaders['train_loader']
        valid_loader = my_loaders['valid_loader']
        epochs = epochs
        steps = 0
        running_loss = 0
        print_every = 40
        model.to(device)
        for e in range(epochs):
            model.train()
            for images, labels in train_loader:
                steps += 1

                optimizer.zero_grad()
                images, labels = images.to(device), labels.to(device)
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    # Make sure network is in eval mode for inference
                    model.eval()

                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        test_loss, accuracy = validation(model, my_loaders['valid_loader'], criterion)

                    print("Epoch: {}/{}.. ".format(e + 1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                          "Validation Loss: {:.3f}.. ".format(test_loss / len(valid_loader)),  # change to valid loader
                          "Validation Accuracy: {:.3f}".format(accuracy / len(valid_loader)))  # change to valid loader

                    running_loss = 0

                    # Make sure training is back on
                    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args('--core-model')
    #TODO learning rate
    #TODO hidden layers
    #TODO epochs
    #TODO choose GPU?  put model on gpu and it will work

