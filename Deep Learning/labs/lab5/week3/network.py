class Network:
    def __init__(self):
        """
        TODO: Initialize a layers attribute
        """
        self.layer_list = []
        self.input_layer = None
        self.output_layer = None
        
    def add(self, layer):
        """
        Adds a new layer to the network.

        Sublayers can *only* be added after their inputs have been added.
        (In other words, the DAG of the graph must be flattened and added in order from input to output)
        :param layer: The sublayer to be added
        """
        # TODO: Implement this method.
        self.layer_list.append(layer)

    def set_input(self,input):
        """
        :param input: The sublayer that represents the signal input (e.g., the image to be classified)
        """
        # TODO: Delete or implement this method. (Implementing this method is optional, but do not
        # leave it as a stub.)
        self.input_layer = input

    def set_output(self,output):
        """
        :param output: SubLayer that produces the useful output (e.g., clasification decisions) as its output.
        """
        # TODO: Delete or implement this method. (Implementing this method is optional, but do not
        # leave it as a stub.)
        #
        # This becomes messier when your output is the variable o from the middle of the Softmax
        # layer -- I used try/catch on accessing the layer.classifications variable.
        # when trying to access read the output layer's variable -- and that ended up being in a
        # different method than this one.
        self.output_layer = output
        
    def forward(self):
        """
        Compute the output of the network in the forward direction.

        :param input: A torch tensor that will serve as the input for this forward pass
        :return: A torch tensor with useful output (e.g., the softmax decisions)
        """
        # TODO: Implement this method
        # TODO: Either remove the input option and output options, or if you keep them, assign the
        #  input to the input layer's output before performing the forward evaluation of the network.
        #
        # Users will be expected to add layers to the network in the order they are evaluated, so
        # this method can simply call the forward method for each layer in order.
        #self.input_layer.set(input)
        for i in self.layer_list:
            print(i)
            i.forward()
            
    def backward(self):
        """
        Compute the gradient of the output of all layers through backpropagation over the 
        gradient tape.

        """
        self.ll = self.layer_list[-1]
        self.ll.grad = tensor(1.0)
        for l in reversed(self.layer_list):
            l.backward()
        
    def step(self, step_size):
        """
        Perform one step of the stochastic gradient descent algorithm
        based on the gradients that were previously computed by backward.

        """
        for l in self.layer_list:
            l.step(step_size)


