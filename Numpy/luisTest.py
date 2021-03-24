# %% Imports
import numpy as np
import network
import pandas as pd
from keras.datasets import mnist


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
     {"input_dim": 28*28, "output_dim": 100, "activation": "relu"},
     {"input_dim": 100, "output_dim": 200, "activation": "relu"},
     {"input_dim": 200, "output_dim": 10, "activation": "softmax"},
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
net = network.network(121312,nn_architecture)
print("Training nekwork")
res = net.trainNetwork(X,Y)


