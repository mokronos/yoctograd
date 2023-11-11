from yoctograd.engine import Value

def test_grad():

    a = Value(1.0)
    b = Value(2.0)
    c = a + b
    d = b + 1
    e = c + d
    e.backward()

    assert a.grad == 1
    assert b.grad == 2
    assert c.grad == 1
    assert d.grad == 1
    assert e.grad == 1
