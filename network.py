import numpy

# Example of input for architecture
# nn_architecture = [
#     {"input_dim": 2, "output_dim": 4, "activation": "relu"},
#     {"input_dim": 4, "output_dim": 6, "activation": "relu"},
#     {"input_dim": 6, "output_dim": 6, "activation": "relu"},
#     {"input_dim": 6, "output_dim": 4, "activation": "relu"},
#     {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
# ]

# Activation Functions

def relu(X):#Faster method
    return (abs(X) + X)/2

class network:

    def __init__(self,seed,architecture):
        np.random.seed(seed)
        # Give a brief rundown of the architecture
        for idx,layer in enumerate(architecture):
            print()
        # Initialize the network
        self.architecture = architecture
        self.weights = {}
        self.biases = {}
        self.num_layers = len(architecture)
        # Construct the whole thing
        for idx, layer in enumerate(architecture):
            out_dim = layer["output_dim"]
            in_dim = layer["input_dim"]
            print("idx is : {}".format(idx))
            self.weights[idx] = np.random.randn(out_dim, in_dim) * 0.1
            self.biases[idx] = np.random.randn(out_dim, 1) * 0.1

        # Done with the network
        return self

    def singleLayerFP(weightMatrix,biasVector,inputMatrix):
        return (weightMatrix@inputMatrix) + biasVector


    # Input:
    #   M features for N samples
    def forward_propagation(self,networkInput):

        curInput = networkInput
        #  Per Layer
        for idx, layer in enumerate(self.architecture):
            actFunction = self.architecture[idx]["activation"]
            curInput = singleLayerFP(self.weights[idx],self.biases[idx],
            curInput = actFunction(curInput)

        networkOutput = curInput

        return networkOutput
                

