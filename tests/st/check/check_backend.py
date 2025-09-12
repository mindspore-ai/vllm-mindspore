import torch
from mrt.torch import backend


def foo(x, y):
    a = torch.reshape(y, (x.shape[1], -1))
    return torch.matmul(x, a)


opt_foo = torch.compile(foo, backend=backend)

x = torch.randn(2, 2)
y = torch.arange(4.0)
bar = foo(x, y)
opt_bar = opt_foo(x, torch.arange(6.0))
opt_bar = opt_foo(x, torch.arange(8.0))
opt_bar = opt_foo(x, y)

assert torch.equal(opt_bar, bar), f"\nopt_bar={opt_bar}\nbar={bar}"
print("The result is correct. 'mrt' backend has been installed successfully.")
