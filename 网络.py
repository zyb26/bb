import torch.cuda
from torch import nn

# hidden1 = nn.ReLU()(hidden1)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # 类
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),  # 类
            nn.Linear(512, 512),  # 类
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

device = (
    "cuda"
    if torch.cuda.is_available()
    else "maps"
    if torch.backends.maps.is_available()
    else "cpu"
)

model = NeuralNetwork().to(device=device)  # 魔术方法 __call__
print(model)

X = torch.rand(1, 28, 28, device=device)
print(X)
logits = model(X)

# 置信度
pred_probab = nn.Softmax(dim=1)(logits)  # 类
# print(nn.Flatten(X))
print(pred_probab)

# softmax 预测概率
softmax = nn.Softmax(dim=1)
prob = softmax(pred_probab)

# 最大的索引
y_pre = prob.argmax(1)
print(y_pre)

# 包含weight和bias；（有时也会打印buffers）;relu层不包含参数不更新
for name, param in model.named_parameters():
    print(f"Layer:{name}|Size:{param.size()}|Values:{param[:2]}")
    