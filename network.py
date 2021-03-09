# %% Imports
import numpy as np
#import network
import pandas as pd
from keras.datasets import mnist

# %% Network

# Example of input for architecture
# nn_architecture = [
#     {"input_dim": 2, "output_dim": 4, "activation": "Relu"},
#     {"input_dim": 4, "output_dim": 6, "activation": "Relu"},
#     {"input_dim": 6, "output_dim": 6, "activation": "relu"},
#     {"input_dim": 6, "output_dim": 4, "activation": "relu"},
#     {"input_dim": 4, "output_dim": 1, "activation": "softmax"},
# ]

# Activation Functions

def Relu(Z):#Faster method
    return (abs(Z) + Z)/2
def dRelu_d(Z):
    ret = np.ones(Z.shape)
    ret[Z<=0] = 0
    return ret
def sigmoid(X):#Faster method
    return 1/(1+np.exp(-X))
def CatCrossEntropy(yhat,y):
    print("catcrossent : yhat.shape={},  y.shape={}".format(yhat.shape,y.shape))
    numSamples = yhat.shape[1]
    logres = np.log(yhat)
    mulres = np.multiply(logres,y)
    addres = (np.sum(mulres,axis=0))*-1
    finalAvg = np.sum(addres)/numSamples
    return finalAvg

# ToCheck
def dCatCrossEntropy_d(a,y):
    return np.divide(y,a)*-1

    #We might have to change this for a stable soft amx
def Softmax(X):
    #Add by row
    shiftX = X-np.max(X)
    exps = np.exp(shiftX)
    return exps/np.sum(exps)
    #print("Applying SoftMax")
    #expMat = np.exp(X)
    #divisor = np.sum(expMat,axis=0)
    #res = expMat / divisor
    #print("shape of softmax divisor res is : {}".format(divisor.shape))
    #return res

def dSoftmax_d(Z):
    a_i = Softmax(Z)
    ret = a

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
        self.memory = {}
        self.weights = {}
        self.num_layers = len(architecture)
        # Construct the whole thing
        for idx, layer in enumerate(architecture):
            out_dim = layer["output_dim"]
            in_dim = layer["input_dim"]
            print("idx is : {} with in {} and out {}".
                    format(idx,in_dim,out_dim))
            self.weights[idx] = np.random.randn(out_dim, in_dim) * 0.01
            self.biases[idx] = np.random.randn(out_dim, 1) * 0.01

        # Done with the network

    def singleLayerFP(self,weightMatrix,biasVector,inputMatrix):
        print("SFP: Shapes : weightmatrix, bias, input : {},{},{}".
                format(weightMatrix.shape,biasVector.shape,inputMatrix.shape));
        return (weightMatrix@inputMatrix) + biasVector
    # Input:
    #   M features for N samples
    def forward_propagation(self,networkInput):
        prevA = networkInput
        #  Per Layer
        for idx, layer in enumerate(self.architecture):
            archi = self.architecture[idx]["activation"]
            if archi == "relu":
                actFunction = Relu
            elif  archi == "sigmoid":
                actFunction = sigmoid
            else:
                print("For softmax we have : {}",format(prevA[0]))
                actFunction = Softmax
            curZ = self.singleLayerFP(self.weights[idx],
                    self.biases[idx],
                    prevA)
            curA = actFunction(curZ)
            self.memory["A"+str(idx)] = curA
            self.memory["Z"+str(idx)] = curZ
            print("FP: output shape : {}".format(curA.shape))
            prevA = curA
        return curA
    
    def perLayer_bwp(self,dLdA,dActFunc,curZ,curA,dZdA,bsize):
        
        
        dLdZ = dLdA@dActFunc(curZ)
        dLdW = dLdZ@curA
        
        dLdb = np.ones((bsize,1))#?
        new_dLdA = dLdZ@dZdA
        
        return new_dldA
    
    def backpropagation(self,Y,A):
        
        cur_dA = dCatCrossEntropy_d(A,Y)
        
        for lidx, layer in reversed(list(enumerate(self.architecture))):
            sdactFunc = "d"+layer["activation"]+"_d"
            dactFunc = dRelu_d
            if sdactFunc == "dRelu_d":
                dactFunc = dRelu_d
            elif sdactFunc == "dSoftmax_d":
                dactFunc = dSoftmax_d
            elif sdactFunc == "dCatCrossEntropy_d":
                dactFunc = dCatCrossEntropy_d
            else:
                print("FATAL: no differential for actication function!")
            #Not sure abotu these prevs
            curA = self.memory["A"+str(lidx)]
            curZ = self.memory["Z"+str(lidx)]
            curW = self.weights[lidx]
            curB = self.biases[lidx]
            
            cur_dA, cur_dW, cur_dB = self.perLayer_bwp(
                cur_dA,dactFunc,curZ,curA,curW,curB.shape[1]
                )
            
            self.grad_weights[lidx] = cur_dW
            self.grad_biases[lidx] = cur_dB
        print("Finished with BackPropagation")
            
            
            


                
    def trainNetwork(self,X,Y):
        #X is input
        print("Doing self propagation")
        Yhat = self.forward_propagation(X)
        print("Yhat shape : {}".format(Yhat.shape))
        print("Done with propagation")
        # Evaluate Loss Function
        loss =  CatCrossEntropy(Yhat,Y)
        self.backpropagation(Y,Yhat)



# %% Read Mnist
def getMnist10():
    #Importing Images
    with open("./train-images-idx3-ubyte","rb") as f:
        magicNum = f.read(4)
        num_images = int.from_bytes(f.read(4),"big")
        num_rows = int.from_bytes(f.read(4),"big")
        num_cols = int.from_bytes(f.read(4),"big")
        imgs = np.zeros((num_images,num_rows,num_cols))
        nBytesTotal = num_images*num_rows*num_cols
        print("This is the num of i,r,c = {},{},{}".format(
            num_images,num_rows,num_cols
            ))
        for idx in range(num_images):
            for i in range(num_rows):
                for j in range(num_cols):
                    imgs[idx][i][j] = int.from_bytes(f.read(1),"big")

        imgs = imgs.reshape(num_images,num_rows*num_cols)
        #plt.imshow(imgs[0],cmap=plt.get_cmap('gray'))
    # Now open up labels
    with open("./train-labels-idx1-ubyte","rb") as f:
        magicNum = int.from_bytes(f.read(4),"big")
        numItems = int.from_bytes(f.read(4),"big")
        labels = np.zeros((1,numItems))
        for idx in range(numItems):
            labels[0][idx] = int.from_bytes(f.read(1),"big",signed=False)

    print(labels[0][0])

    return imgs,labels
        


nn_architecture = [
     {"input_dim": 28*28, "output_dim": 100, "activation": "Relu"},
     {"input_dim": 100, "output_dim": 200, "activation": "Relu"},
     {"input_dim": 200, "output_dim": 10, "activation": "Softmax"},
     ]

# %% Importing data

# print("Importing Data...")
# X,Y  = getMnist10()

(X, Y), (X_test, y_test) = mnist.load_data()
X = X.reshape(60000,28*28)
X = X/255
s = pd.Series(Y)
Y = pd.get_dummies(s)
Y = np.transpose(Y).to_numpy()
X = np.transpose(X)

# %% Network training
net = network(121312,nn_architecture)
print("Training nekwork")
res = net.trainNetwork(X,Y)


