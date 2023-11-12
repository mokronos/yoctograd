from yoctograd.engine import Value
import random


class Module():

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return out

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer({len(self.neurons)})"


class MLP(Module):
    def __init__(self, nin, nouts):
        self.layers = [Layer(nin, nouts[0])]
        self.layers.extend(Layer(nouts[i], nouts[i + 1])
                           for i in range(len(nouts) - 1))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP({[len(layer.neurons) for layer in self.layers]})"
