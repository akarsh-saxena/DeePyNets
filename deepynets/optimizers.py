import numpy as np

class GradientDescent():

    def __init__(self, learning_rate=0.01, **kwargs):
        self.learning_rate = learning_rate

    def __call__(self, model, **kwargs):
        self.update(model)

    def update(self, model):
        for i in range(model.total_layers):
            new_W = model.get_config(i, 'W') - self.learning_rate * model.get_config(i, 'dW')
            new_b = model.get_config(i, 'b') - self.learning_rate * model.get_config(i, 'db')
            model.set_config(i, 'W', new_W)
            model.set_config(i, 'b', new_b)


class GradientDescentWithMomentum():

    def __init__(self, learning_rate=0.01, beta=0.9, **kwargs):
        self.learning_rate = learning_rate
        self.beta = beta

    def __call__(self, model, **kwargs):
        self.update(model)

    def update(self, model):
        for i in range(model.total_layers):
            new_VdW = self.beta * model.get_config(i, 'VdW') + (1-self.beta * model.get_config(i, 'dW'))
            new_Vdb = self.beta * model.get_config(i, 'Vdb') + (1 - self.beta * model.get_config(i, 'db'))
            model.set_config(i, 'VdW', new_VdW)
            model.set_config(i, 'Vdb', new_Vdb)

            new_W = model.get_config(i, 'W') - self.learning_rate * model.get_config(i, 'VdW')
            new_b = model.get_config(i, 'b') - self.learning_rate * model.get_config(i, 'Vdb')
            model.set_config(i, 'W', new_W)
            model.set_config(i, 'b', new_b)
