import torch
import torch.nn as nn
# Module类常用属性和方法

# 对特定层进行权重初始化
@torch.no_grad()
def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        m.weight.fill_(1.0)
        print(m.weight)
net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(init_weights)

# 增加层数 --> 加入到了self._modules字典中
net.add_module(name="linear3", module=nn.Linear(2, 2))
print(net)

# buffers 和parameter 均参与梯度下降
for buf in net.buffers():  # net.named_buffers()
    print(type(buf), buf.size())

# chidren() 返回所有的子模块
print(net.children())

# cpu()/cuda() 把模型和参数都搬到gpu上
net.cuda()

# eval()/train() 主要区别dropout/batchnormal运行逻辑

# get_parameter() 根据一个字符串得到当前模型的一个参数

# get_submodule() 根据一个模型参数获取其子模块

# load_state_dict(state_dict, strict=True) 模型加载参数
# state_dict
# optimizer = nn.
# torch.save({'model_state_dict': net.state_dict,
#             'optimizer_state_dict':optimizer.state_dict}, PATH)

# model = Net()
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# check_point = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict']
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch'] # 在原来的基础上进行 ++
# loss = checkpoint['loss']
# model.eval()
# model.train()

# requires_grad_() 是否参与梯度更新  model.parameters.required_grad = True

# 父类方法nn.Module源码（类方法，类属性）
"""
1.默认 self.training = True
注册函数
2.register_buffer # 可以和parameters 一起保存下来 state_dict --> eval（）时候使用
class BatchNorm(nn.Module):
    def __init__(self)
    self.register_buffer('running_mean', torch.zeros(num_features))
    self.register_buffer('running_vars', torch.zeros(num_features))
3.register_parameters() 往模块中添加一个参数,参数就可以通过他的名称进行获取 --> 添加到self.parameters()字典中
nn.parameter.Parameter(data=tensor, requires_grad=True)  # Parameter这个类 --> 自动加入到parameters（）中
实例：
    self.register_parameters("mean", nn.Parameter(torch.zeros(1), requires_grad=True)
    等价于self.mean = nn.Parameter(torch.zeros(1), requires_grad=True)
    self.state_dict()["mean"]
    register_module也是同样的
4.get_submodule(A.net_b.net_c) 根据某种特定字符串获取子Module模块（层/block）--> 参数/requires_grad
5.get_parameter()根据一个字符串得到一个参数
hasattr(mod, param_name)判断一个对象中是否有这个属性
param:torch.nn.Parameter = getattr(mod, param_name)  # 获取这个模型的属性赋值给param(属于Parameter这个类)
判断param是否属于torch.nn.Parameter这个类 --> isinstance(param, torch.nn.Parameter)

6.get_buffer()状态变量 滑动均值/方差
module_path, _, buffer_name = traget.rpartion(".")
mod = self.get_submodule(module_path)
buffer = getattr(mod, buffer_name) --> mod.buffer_name
if buffer_name not in mod._buffers:
    raise AttributeError("`" + buffer_name + "`is not a buffer")
7.apply
for module in self.children():
    module.apply(fn) 所有子模块传到方法中
fn(self) # 把自己传进去
return self
8.cuda --> TORCH.TENSOR.CUDA(COPY到cuda内存中) 背后都是调用的self._apply函数把模型 参数 buffer逐一复制到cuda
lamda t:t.cuda(device)
9.type函数
把模型的参数和buffer都转换一个类型
10.数据格式函数 （float, double, half, bfloat16, to_empty）
model.double() --> 将所有参数 buffer转换为double类型
self._apply(lamda t:torch.empty_like(t, device=device))
"""



