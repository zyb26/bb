import torch
import torch.nn as nn
model = nn.Sequential(
    nn.Conv2d(1, 20, 5),
    nn.ReLU(),
    nn.Conv2d(20, 64, 5),
    nn.ReLU()
)

# model = nn.Sequential(
#     ('conv1', nn.Conv2d(1, 20, 5)),
#     ('relu1', nn.ReLU()),
#     ('conv2', nn.Conv2d(20, 64, 5)),
#     ('relu2', nn.ReLU())
# )

# # 如果是字典
# self.add_model(key, module)
# # 如果不是 key 就是0 1 2 3 4 5 ...
# for idx, module in enumerate(args):
#     self.add_model(str(idx), module)
# forward(self, input):
#     for module in self:
#         input = module(input)
#         return module(input)

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.choices = nn.ModuleDict({
            'conv': nn.Conv2d(10, 10, 3),
            'pool': nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['prelu', nn.PReLU()]
        ])

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

    def forward(self, x):
        for i, p in enumerate(self.params):
            x = self.params[i//2].mm(x) + p.mm(x)
        return x

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.params = nn.ParameterDict({
            'left': nn.Parameter(torch.randn(5, 10)),
            'right': nn.Parameter(torch.randn(5, 10))
        })

    def forward(self, x, choice):
        x = self.params[choice].mm(x)
        return x