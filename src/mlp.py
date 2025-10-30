class MLP():

    # TODO: HERE!!!!!!!!!!!!!!!
    def __init__(self, 
                 activation_func: function, 
                 num_hidden_layers: int, 
                 hidden_layer_size: int
                 ) -> Any:
        # Initialize layers, weights, biases, etc.
        pass

    def fit(self, X, y, learning_rate: float, epochs: int):
        # Train the MLP on data x with labels y
        pass

    def predict(self, x):
        # Predict output for input x
        pass

    # Optional but probably useful methodes below...

    def forward(self, x):
        # Define forward pass
        pass

    def backward(self, grad_output):
        # Define backward pass
        pass

    def update_parameters(self, learning_rate):
        # Update weights and biases
        pass