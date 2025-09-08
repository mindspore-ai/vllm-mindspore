import torch
from mrt.torch import backend


def foo(x, y):
    a = torch.add(x, y)
    b = torch.sub(x, y)
    return a * b


opt_foo = torch.compile(foo, backend=backend)

x = torch.randn(2, 2)
y = torch.randn(2, 2)
bar = foo(x, y)
opt_bar = opt_foo(x, y)

assert torch.equal(opt_bar, bar), f"\nopt_bar={opt_bar}\nbar={bar}"
print("The result is correct. 'mrt' backend has been installed successfully.")
