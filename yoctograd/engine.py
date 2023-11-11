class Value():
    def __init__(self, data, children=(), op='', label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self.children = children
        self.op = op
        self.label = label

    def __repr__(self):
        return f"Value({self.label + '| ' if self.label else ''}data={self.data}, grad={self.grad})"

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        self._backward = _backward
        return out

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        self._backward = _backward
        return out

    def __pow__(self, other):
        return self ** other

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def backward(self):

        graph = []

        def build_graph(value):
            graph.append(value)
            for child in value.children:
                build_graph(child)

        build_graph(self)

        self.grad = 1
        for value in graph:
            value._backward()


if __name__ == "__main__":
    pass
    x = Value(1.0, label='x')
    w = Value(2.0, label='w')
    b = Value(3.0, label='b')

    y = w * x + b

    y.backward()
