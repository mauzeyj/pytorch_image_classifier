class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_out=0.5):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)


def load_transfer_model(version):#TODO add ability to pick transfer model
    model = models.vgg16(pretrained = True)

    for param in model.paramaters():
        param.requires_grad = False
    return model