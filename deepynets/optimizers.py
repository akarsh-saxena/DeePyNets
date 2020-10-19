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


class Adam():

    def __init__(self, learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-7, **kwargs):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iteration = 0
        self.corrected_lr = learning_rate

    def __call__(self, model, **kwargs):
        self.update(model)
        self.iteration = kwargs['epoch'] + 1
        self.batch_size = kwargs['batch_size']

    def update(self, model):

        for i in range(model.total_layers):
            new_VdW = self.beta_1 * model.get_config(i, 'VdW') + (1 - self.beta_1) * model.get_config(i, 'dW')
            new_Vdb = self.beta_1 * model.get_config(i, 'Vdb') + (1 - self.beta_1) * model.get_config(i, 'db')
            model.set_config(i, 'VdW', new_VdW)
            model.set_config(i, 'Vdb', new_Vdb)

            new_SdW = self.beta_2 * model.get_config(i, 'SdW') + (1 - self.beta_2) * model.get_config(i, 'dW')**2
            new_Sdb = self.beta_2 * model.get_config(i, 'Sdb') + (1 - self.beta_2) * model.get_config(i, 'db')**2
            model.set_config(i, 'SdW', new_SdW)
            model.set_config(i, 'Sdb', new_Sdb)

            self.corrected_lr = self.learning_rate * np.sqrt(1-self.beta_2**self.iteration)/(1-self.beta_1**self.iteration)

            new_W = model.get_config(i, 'W') - self.corrected_lr* model.get_config(i, 'VdW') / \
                    np.sqrt(model.get_config(i, 'SdW')+self.epsilon)

            new_b = model.get_config(i, 'b') - self.corrected_lr * model.get_config(i, 'Vdb') / \
                    np.sqrt(model.get_config(i, 'Sdb')+self.epsilon)

            model.set_config(i, 'W', new_W)
            model.set_config(i, 'b', new_b)
    