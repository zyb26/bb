import torch
import torchvision.models as models

# 这样保存仅能用来做推理了（训练的时候不光要保存模型的权重还要保存优化器的状态）
# model = models.vgg16(pretrained=True)
# torch.save(model.state_dict(), 'model_weights.pth')
#
# model = models.vgg16()
# model.load_state_dict(torch.load('model_weights.pth'))
# model.eval()
import torch.nn as nn
class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 3)
        self.linear2 = nn.Linear(3, 4)
        self.batch_norm = torch.nn.BatchNorm2d(4)

test_module = Test()
# print(test_module.linear1)
print(test_module._modules)
print(test_module._modules['linear1'])
print(test_module._modules['linear1'].weight)
print(test_module._modules['linear1'].weight.dtype)

# to函数 将module中所有的浮点类型的数据都变成double
test_module.to(torch.double)
print(test_module._modules['linear1'].weight.dtype)

# __getattr__ 魔法函数 中的三个属性
# _modules  OrderedDict([('linear1', Linear(in_features=2, out_features=3, bias=True)), ('linear2', Linear(in_features=3, out_features=4, bias=True)), ('batch_norm', BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))])
print(test_module._modules)
# _parameters OrderedDict() 只遍历了模型没遍历子模块 --> 没有nn.parameter 但是有参数的
print(test_module._parameters)
#_buffers
print(test_module._buffers)

# net.state_dict()
# self._save_to_state_dict() 当前模型的参数都存到字典中
# for name, module in self._modules.items()
#    if module is not None:
#       module.state_dict() 递归调用 _save_to_state_dict

# print(test_module.state_dict())  # w b w b batch_norm的w b 滑动平均 var

test_module.state_dict()['linear1.weight']

# load_state_dict
# 得到自身buffer parameters 放到local_state
# 遍历name, param in local_state
#     key = prefix + name
#     if key in state_dict:
#         input_param = state_dict[key]

# parameters 内部是调用named_parameters  (state_dict内部有buffer参数)
# for name, param in self.named_parameters(recurse=recurse)
#       yield param
for p in test_module.parameters():  # 是一个生成器
    print(p)
for p in test_module.parameters():  # 不仅包含张量的值还具体到属于哪一个模块
    print(p)

# children 函数 与 named_children函数
# for name, module in self.named_children():
#   yield module

# 返回的是一个迭代器同样用一个循环来调用他
# for name, module in self._modules.item():
#   if module is not None and module not in memo:
#       memo.add(module)
#       yield name, module

# modules named_modules 会返回多个 一个总的 然后每个子模块的

