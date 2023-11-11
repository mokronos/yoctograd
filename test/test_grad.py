from yoctograd.engine import Value
import torch


def test_grad_manual():

    a = Value(1.0)
    b = Value(2.0)
    c = a + b
    d = b + 1
    e = c * d
    e.backward()
    # e = (a + b) * (b + 1)
    # e = ab + b**2 + a + b

    # forward pass
    assert e.data == 9.0
    # backward pass
    assert a.grad == 3
    assert b.grad == 6
    assert c.grad == 3
    assert d.grad == 3
    assert e.grad == 1


def test_grad_torch():

    a = Value(1.0)
    b = Value(2.0)
    c = a + b
    d = b + 1
    e = c * d
    e.backward()

    ayg, byg, eyg = a, b, e

    a = torch.tensor([1.0]).double()
    b = torch.tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = b + 1
    e = c * d
    e.backward()

    at, bt, et = a, b, e

    tol = 1e-6
    # forward pass
    assert abs(eyg.data - et.data) < tol
    # backward pass
    assert abs(ayg.grad - at.grad) < tol
    assert abs(byg.grad - bt.grad) < tol


def test_more_ops():

    a = Value(1.0)
    b = Value(2.0)
    c = a + b
    d = b * a + a**8
    e = c * 2
    f = e - d.relu()
    f += 9 * b * (-a).relu()
    g = f / 2
    g += 5 / e
    h = g**3
    h.backward()

    ayg, byg, hyg = a, b, h

    a = torch.tensor([1.0]).double()
    b = torch.tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = b * a + a**8
    e = c * 2
    f = e - d.relu()
    f += 9 * b * (-a).relu()
    g = f / 2
    g += 5 / e
    h = g**3
    h.backward()

    at, bt, ht = a, b, h

    tol = 1e-6
    # forward pass
    assert abs(hyg.data - ht.data) < tol
    # backward pass
    assert abs(ayg.grad - at.grad) < tol
    assert abs(byg.grad - bt.grad) < tol


def test_relu():

    a = Value(-1.0)
    b = a.relu()
    b.backward()

    a = torch.tensor([-1.0]).double()
    a.requires_grad = True
    b = a.relu()
    b.backward()

    tol = 1e-6

    # forward pass
    assert abs(b.data - b.data) < tol
    # backward pass
    assert abs(a.grad - a.grad) < tol
