import torch.nn.functional as F
from torch import nn
from torchvision import models


class Network(nn.Module):
    def __init__(self, input, output, hidden_layers, drop_out=0.5):
        """
        takes inputs for 2 layer neural network and creates a pytorch model
        :param input: input size of first input later
        :param output: output size of prediction layer
        :param hidden_layers: 2 layer sizes
        :param drop_out: amount of dropout for the network
        """
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(hidden1, hidden2) for hidden1, hidden2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        """
        Network forward pass over the data to get the output
        :param x: data to be passed over
        :return: 1d tensor ready for back propagation or softmax
        """
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)


def load_transfer_model(version):
    """
    Loads the one of the vgg models
    :param version: version of vgg available from pytorch.models
    :return: pretrained vgg model
    """
    version = version.lower()
    if version == 'vgg11':
        model = models.vgg11(pretrained = True)
    elif version == 'vgg13':
        model = models.vgg13(pretrained = True)
    elif version == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif version == 'vgg19':
        model = models.vgg18(pretrained = True)
    else:
        print('I am only familiar with vgg11, vgg13, vgg16 or vgg19, please pick one of those.')
    for param in model.parameters():
        param.requires_grad = False
    return model


def rebuild_full_model(classifier, transfer_model):
    model = transfer_model
    model.classifier = classifier
    return model