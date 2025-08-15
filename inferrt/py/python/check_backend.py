import torch
from dapy import backend


def foo(x, y):
    a = torch.add(x, y)
    b = torch.sub(x, y)
    return a * b


opt_foo = torch.compile(foo, backend=backend)

x = torch.randn(2, 2)
y = torch.randn(2, 2)
assert torch.equal(opt_foo(x, y), foo(x, y))
print("The result is correct. 'dapy' backend has been installed successfully.")
