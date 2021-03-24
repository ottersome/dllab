# %% Imports
import numpy as np
#import network
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist

# %% Network

# Example of input for architecture
# nn_architecture = 
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
def CatCrossEntropy(yhat,y,epsilon=1e-12):
    #print("catcrossent : yhat.shape={},  y.shape={}".format(yhat.shape,y.shape))
    numSamples = yhat.shape[1]
    yhat = np.clip(yhat, epsilon, 1. - epsilon)
    logres = np.log(yhat+0.000001)
    mulres = np.multiply(logres,y)
    addres = (np.sum(mulres,axis=0))*-1
    finalAvg = np.sum(addres)/numSamples
    return finalAvg

# ToCheck
def dCatCrossEntropy_d(a,y):
    ret = []
    for i in range(a.shape[1]):
        ret.append(np.divide(y,a))
    return np.divide(y,a)*-1

    #We might have to change this for a stable soft amx
def Softmax(Z):
    #Add by row
    #shiftZ = Z-np.amax(Z)
    #exps = np.exp(shiftZ)
    #return exps/np.sum(exps,axis=0)
    #print("Applying SoftMax")
    expMat = np.exp(Z)
    divisor = np.sum(expMat,axis=0,keepdims=True)
    res = expMat / divisor
    #print("shape of softmax divisor res is : {}".format(divisor.shape))
    return res

def dSoftmax_d(Z):
    pass
    #zSoft = Softmax(Z)
    #dA_dZ = np.ndarray(Z.shape)
    #dAdZ = np.diag((10,Z.shape[0]))
    #s = Softmax(Z)
    #ret = []
    #for col_idx in range(Z.shape[1]):#FOr every sample
    #    col = s[:,col_idx].reshape(-1,1)
    #    #jacmat = np.diagflat(col) - np.dot(col, col.T)
    #    jacmat = np.diagflat(col) - np.dot(col, col.T)
    #    ret.append(jacmat)
    #return ret

def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    squeezo = np.squeeze(cost)
    return np.squeeze(cost)

class network:

    def __init__(self,seed,architecture):
        np.random.seed(seed)
        # Give a brief rundown of the architecture

        # Initialize the network
        self.architecture = architecture
        self.weights = {}
        self.biases = {}
        self.memory = {}
        self.weights = {}
        self.grad_weights = {}
        self.grad_biases = {}
        self.epochs = 100
        self.learningRate = 0.1
        self.num_layers = len(architecture)
        # Construct the whole thing
        for idx, layer in enumerate(architecture):
            out_dim = layer["output_dim"]
            in_dim = layer["input_dim"]
            print("idx is : {} with in {} and out {}".
                    format(idx,in_dim,out_dim))
            print("\tSo dimensions are {},{}".format(out_dim,in_dim))
            self.weights[idx] = np.random.normal(loc=0.0, scale= np.sqrt(2/(out_dim+in_dim)),size=(out_dim, in_dim)) #* 0.1
            self.biases[idx] = np.random.randn(out_dim, 1) * 0.1

        # Done with the network

    def singleLayerFP(self,weightMatrix,biasVector,inputMatrix):
        return np.dot(weightMatrix,inputMatrix) + biasVector
    # Input:
    #   M features for N samples
    def forward_propagation(self,networkInput):
        prevA = networkInput
        self.memory["A-1"] = prevA
        #  Per Layer
        for idx, layer in enumerate(self.architecture):
            archi = self.architecture[idx]["activation"]
            if archi == "Relu":
                actFunction = Relu
            elif  archi == "Sigmoid":
                actFunction = sigmoid
            else:
                actFunction = Softmax
            curZ = self.singleLayerFP(self.weights[idx],
                    self.biases[idx],
                    prevA)
            curA = actFunction(curZ)
            self.memory["A"+str(idx)] = curA
            self.memory["Z"+str(idx)] = curZ
            prevA = curA
        return curA
    
    def perLayer_bwp(self,dLdA,dActFunc,curZ,prevA,dZdA,bsize,lidx):
        
        m = dLdA.shape[1]
        # Last layer should be softmax
        if lidx == 2:
            #Then we do both Loss and Softmax at once
            dLdZ = dLdA 
        else:    
            dLdZ = np.multiply(dLdA,dActFunc(curZ))
        dLdW = np.dot(dLdZ,prevA.T) 
        
        dLdB = np.sum(dLdZ, axis=1, keepdims=True) 
        new_dLdA = np.dot(dZdA.T,dLdZ)
        
        return new_dLdA,dLdW,dLdB
    
    def backpropagation(self,Y,A):
       
        cur_dA = (A - Y)/Y.shape[1]#Derivative of CatCrossEntr and SoftMax together
        
        for lidx, layer in reversed(list(enumerate(self.architecture))):
            sdactFunc = "d"+layer["activation"]+"_d"
            dactFunc = dRelu_d
            if sdactFunc == "dRelu_d":
                dactFunc = dRelu_d
            elif sdactFunc == "dSoftmax_d":
                dactFunc = dSoftmax_d
            else:
                print("FATAL: no differential for actication function!")
            #Not sure abotu these prevs

            prevA = self.memory["A"+str(lidx-1)]
            curZ = self.memory["Z"+str(lidx)]
            curW = self.weights[lidx]
            curB = self.biases[lidx]
            
            cur_dA, cur_dW, cur_dB = self.perLayer_bwp(
                cur_dA,dactFunc,curZ,prevA,curW,curB.shape[1],lidx)
            
            self.grad_weights[lidx] = cur_dW
            self.graBd_biases[lidx] = cur_dB
                   
    def updateParams(self):
        for lidx, layer in enumerate(self.architecture):
            #plt.hist(self.weights[lidx].flatten(),bins="auto",label="Weights at {} before update".format(lidx))
            self.weights[lidx] -= self.learningRate * self.grad_weights[lidx]
            #plt.hist(self.weights[lidx].flatten(),bins="auto",label="Weights at {} after update".format(lidx))
            #plt.show()
            self.biases[lidx] -= self.learningRate * self.grad_biases[lidx]

                
    def trainNetwork(self,X,Y):
        #X is input
        lossPerEpoch = []
        accPerEpoch = []
        for idx,epoch in enumerate(range(self.epochs)):
            Yhat = self.forward_propagation(X)
            eqs = np.sum(np.argmax(Yhat,axis=0) == np.argmax(Y,axis=0))
            accPerEpoch.append(eqs/Y.shape[1])
            # Evaluate Loss Function
            
            lossPerEpoch.append(CatCrossEntropy(Yhat,Y))
            self.backpropagation(Y,Yhat)
            self.updateParams()
            print("At epoch {} we get a loss of {} with accuracy of : {}".format(epoch,lossPerEpoch[idx],accPerEpoch[idx]))
        print("Now displaying charts")
        return lossPerEpoch, accPerEpoch




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
Y = np.array(np.transpose(Y).astype("float"))
X = np.array(np.transpose(X).astype("float"))

# %% Network training
net = network(42069,nn_architecture)
print("Training nekwork")
lpe, ape = net.trainNetwork(X,Y)

fig = plt.figure()
plt.plot(range(100),lpe,label="Loss Per Epoch")
plt.plot(range(100),ape,label="Accuracy Per Epoch")
plt.show()

