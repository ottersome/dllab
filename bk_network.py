import numpy as np 

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
def sigmoid(X):#Faster method
    return 1/(1+np.exp(-X))
def 

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
            print("idx is : {} with in {} and out {}".
                    format(idx,in_dim,out_dim))
            self.weights[idx] = np.random.randn(out_dim, in_dim) * 0.1
            self.biases[idx] = np.random.randn(out_dim, 1) * 0.1

        # Done with the network

    def singleLayerFP(self,weightMatrix,biasVector,inputMatrix):
        print("SFP: Shapes : weightmatrix, bias, input : {},{},{}".
                format(weightMatrix.shape,biasVector.shape,inputMatrix.shape));
        return (weightMatrix@inputMatrix) + biasVector
    # Input:
    #   M features for N samples
    def forward_propagation(self,networkInput):
        curInput = networkInput
        #  Per Layer
        for idx, layer in enumerate(self.architecture):
            archi = self.architecture[idx]["activation"]
            if archi == "relu":
                actFunction = relu
            else:
                actFunction = sigmoid
            curOutput = self.singleLayerFP(self.weights[idx],
                    self.biases[idx],
                    curInput)
            curOutput = actFunction(curOutput)
            print("FP: output shape : {}".format(curOutput.shape))
            curInput = curOutput
        return curInput
                
    def trainNetwork(self,X):
        #X is input
        print("Doing self propagation")
        Yt = self.forward_propagation(X)
        print("Done with propagation")
        # Evaluate Loss Function

        


