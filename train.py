import argparse

import torch
from functions import get_device, standard_folders_config, transformations, loaders, validation
from torch import nn
from torch import optim

from models import Network, load_transfer_model
from functions import validation, transformations


def train(hidden_layers, epochs, drop_out = 0.5, core_model ='vgg16'):
    """
    train a pytorch model
    :param hidden_layers: list of 2 int, numbers for 2 layers
    :param epochs: int, number of epochs to train model
    :param drop_out: float, dropout
    :param core_model: pytorch pretrained vgg model, model to use for transfer learning
    :return: trained pytorch model
    """
    folders = standard_folders_config()
    classifier = Network(25008, 102, hidden_layers, drop_out)
    model = load_transfer_model(core_model)
    model.classifier = classifier
    my_loaders = loaders(folders['train_dir'],
                         folders['valid_dir'],
                         folders['test_dir'],
                         transformations())
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

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
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validation(model, my_loaders['valid_loader'], criterion)
                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss / len(valid_loader)),  # change to valid loader
                      "Validation Accuracy: {:.3f}".format(accuracy / len(valid_loader)))  # change to valid loader
                running_loss = 0
                model.train()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--core-model', default = 'vgg16')
    parser.add_argument('--hidden-layers', default = [10000, 1000], nargs='+', type = int)
    parser.add_argument('--learning-rate', default=0.001)
    parser.add_argument('--epochs', default = 10, type = int)
    parser.add_argument('--drop-out', default = .5, type = float)
    args = parser.parse_args()
    args = args.__dict__
    model = train(args['hidden_layers'], args['epochs'], args['drop_out'], core_model=args['core_model'])
    model_checkpoint = {'transfer_learning_model':args['core_model'],
                   'input_size': 25088,
                   'output_size': 102,
                   'hidden_layers': args['hidden_layers'],
                   'drop_out':args['drop_out'],
                   'state_dict':model.classifier.state_dict()}
    torch.save(model_checkpoint, 'checkpoint.pth')




