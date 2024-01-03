# train()
# self.training = True
# for module in self.children():
# module.train(mode)
# return self
# dropout 的父类也是nn.Module 然后drop属于这个网络子模块 子模块也设置了self.training = True 于是就生效了 调用父类的self.training
# dropout --> return F.dropout(input, self.p, self.training, self.inplace)

# batch_norm 继承了module
# if self.training:

# eval()
# return self.train(False)

# requires_grad_(False)
# 模型所有参数requires_grad_变为False

# zero_grad 梯度清零 优化器继承的这个
# str(test_module)
# dir(test_module)
# self.__class__
# self.__dict__
# self._parameters.keys()
# self._modules.keys()
# self._buffer.keys()
