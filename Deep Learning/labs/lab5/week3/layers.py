import numpy as np
import torch

# TODO: Please be sure to read the comments in the main lab and think about your design before
# you begin to implement this part of the lab.

# Layers in this file are arranged in roughly the order they
# would appear in a network.


class Layer:
    def __init__(self, output_shape):
        """
        TODO: Add arguments and initialize instance attributes here.
        """
        self.output_shape = output_shape
        self.grad = None
        self.train = True
        self.output = None
        
    def accumulate_grad(self, new_grad):
        """
        TODO: Add arguments as needed for this method.
        This method should accumulate its grad attribute with the value provided.
        """
        if self.grad == None:
          self.grad = torch.zeros(new_grad.shape)
          self.grad = new_grad

    def clear_grad(self):
        """
        TODO: Add arguments as needed for this method.
        This method should clear grad elements. It should set the grad to the right shape 
        filled with zeros.
        """
        self.grad = torch.zeros(self.output)
        
    def step(self):
        """
        TODO: Add arguments as needed for this method.
        Most tensors do nothing during a step so we simply do nothing in the default case.
        """
        pass

class Input(Layer):
    def __init__(self, output, train=False):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, output_shape=output) # TODO: Pass along any arguments to the parent's initializer here
        self.output = output
        self.train = train
        
    def set(self, value):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set the output of this input layer.
        :param output: The output to set, as a torch tensor. Raise an error if this output's size
                       would change.
        """
        if isinstance(self.output_shape, int):
            assert self.output_shape == value.shape[0]
        else:
            assert self.output_shape == value.shape
        self.output = value

    def randomize(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set the output of this input layer to random values sampled from the standard normal
        distribution (torch has a nice method to do this). Ensure that the output does not
        change size.
        """
        self.output = torch.randn(self.output, dtype=dtype)

    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        pass

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        This method does nothing as the Input layer should have already received its output
        gradient from the previous layer(s) before this method was called.
        """
        pass

    def step(self, step_size):
        """
        TODO: Add arguments as needed for this method.
        This method should have a precondition that the gradients have already been computed
        for a given batch.

        It should perform one step of stochastic gradient descent, updating the weights of
        this layer's output based on the gradients that were computed and a learning rate.
        """
        if self.train:
            print(self.grad.shape)
            self.output -= step_size * self.grad
            print(self.output.shape)

class Linear(Layer):
    def __init__(self, features, weight, bias):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, features.output) # TODO: Pass along any arguments to the parent's initializer here.
        self.features = features
        self.weight = weight
        self.bias = bias
        #self.W_grad = None
        #self.B_grad = None
        
    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        sum_linear = torch.matmul(self.weight.output, self.features.output) + self.bias.output
        #print(sum_linear)
        self.output = sum_linear

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        if self.weight.train:
          print(self.grad)
          dj_dw = self.grad @ self.features.output.T
          print(dj_dw)
          self.weight.accumulate_grad(dj_dw)
        
        #Fix this equation 
        if self.features.train:
          dj_dx = self.grad @ self.weight.output.T
          self.features.accumulate_grad(dj_dx)

        dj_db = self.grad
        self.bias.accumulate_grad(dj_db)

    def step(self, step_size):
        self.weight.output -= step_size * self.weight.grad

    
class ReLU(Layer):
    def __init__(self, previous_layer):
        """
        TODO: Accept any arguments specific to this child class.
        """
        self.previous_layer = previous_layer
        Layer.__init__(self, output_shape=previous_layer.output_shape) # TODO: Pass along any arguments to the parent's initializer here.
    
    def forward(self):
        """  
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        self.output = self.previous_layer.output * (self.previous_layer.output > 0)
        #print(self.output)

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        #print(self.previous_layer.output)
        #print(self.grad)
        #print(self.previous_layer.train)
        if self.previous_layer.train:
            x = self.grad * (self.previous_layer.output > 0)
            self.previous_layer.accumulate_grad(x)


class MSELoss(Layer):
    """
    This is a good loss function for regression problems.

    It implements the MSE norm of the inputs.
    """
    def __init__(self, previous_layer, true):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, 1) # TODO: Pass along any arguments to the parent's initializer here.
        
        self.previous_layer = previous_layer
        self.true = true
        
    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        self.output = self.grad = (1/len(self.previous_layer.output)) * torch.sum(torch.square(self.previous_layer.output - self.true.output))

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        if self.previous_layer.train:
            self.previous_layer.accumulate_grad(2 / len(self.previous_layer.output) * (self.previous_layer.output - self.true.output) * self.grad)


class Regularization(Layer):
    def __init__(self, previous_layer, regularizationConst):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, 1) # TODO: Pass along any arguments to the parent's initializer here.
        self.previous_layer = previous_layer
        self.regularizationConst = regularizationConst
        
    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        self.output = (self.regularizationConst/2)*((self.previous_layer.weight.output.norm()**2))
        #print(self.output)

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        grad = self.regularizationConst * self.previous_layer.weight.output
        #print(grad)
        #print(self.previous_layer.grad)
        self.previous_layer.weight.accumulate_grad(grad)
        #print(self.previous_layer.grad)

 
class Softmax(Layer):
    """
    This layer is an unusual layer.  It combines the Softmax activation and the cross-
    entropy loss into a single layer.

    The reason we do this is because of how the backpropagation equations are derived.
    It is actually rather challenging to separate the derivatives of the softmax from
    the derivatives of the cross-entropy loss.

    So this layer simply computes the derivatives for both the softmax and the cross-entropy
    at the same time.

    But at the same time, it has two outputs: The loss, used for backpropagation, and
    the classifications, used at runtime when training the network.

    TODO: Create a self.classifications property that contains the classification output,
    and use self.output for the loss output.

    See https://www.d2l.ai/chapter_linear-networks/softmax-regression.html#loss-function
    in our textbook.

    Another unusual thing about this layer is that it does NOT compute the gradients in y.
    We don't need these gradients for this lab, and usually care about them in real applications,
    but it is an inconsistency from the rest of the lab.
    """
    def __init__(self, previous_layer, y_true, epsilon=None):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, y_true.output) # TODO: Pass along any arguments to the parent's initializer here.
        self.previous_layer = previous_layer
        self.y_true = y_true
        #self.classif = None
        self.epsilon = epsilon
        
    #cross entropy and softmax for the forward pass
    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        x = self.previous_layer.output.float()
        f = x - torch.max(x)
        log = f - torch.log(torch.sum(torch.exp(f)))
        prob = self.y_true.output * log
        self.output = -torch.sum(prob) / self.y_true.output.shape[0]
        
    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        if self.previous_layer.train:
            x = self.previous_layer.output - self.y_true.output
            self.previous_layer.accumulate_grad(x)


class Sum(Layer):
    def __init__(self, l1, l2):
        """
        TODO: Accept any arguments specific to this child class.
        """
        Layer.__init__(self, 1) # TODO: Pass along any arguments to the parent's initializer here.
        self.l1 = l1
        self.l2 = l2
        
    def forward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: Set this layer's output based on the outputs of the layers that feed into it.
        """
        l1_o = self.l1.output
        l2_o = self.l2.output
        self.output = l1_o + l2_o

    def backward(self):
        """
        TODO: Accept any arguments specific to this method.
        TODO: This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        if self.l1.train:
            self.l1.accumulate_grad(self.grad)
        if self.l2.train:
            self.l2.accumulate_grad(self.grad)
