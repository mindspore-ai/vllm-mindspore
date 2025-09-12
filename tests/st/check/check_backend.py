import torch
from mrt.torch import backend


def foo(x, y, z):
    a = torch.reshape(y, z.shape)
    return torch.matmul(x, a)


opt_foo = torch.compile(foo, backend=backend)

x = torch.randn(2, 2)
y = torch.arange(4.0)
z = torch.randn(2, 2)
bar = foo(x, y, z)
opt_bar = opt_foo(x, torch.arange(6.0), torch.randn(2, 3))
opt_bar = opt_foo(x, torch.arange(8.0), torch.randn(2, 4))
opt_bar = opt_foo(x, y, z)

assert torch.equal(opt_bar, bar), f"\nopt_bar={opt_bar}\nbar={bar}"
print("The result is correct. 'mrt' backend has been installed successfully.")
