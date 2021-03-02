import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline 



# %% Read Mnist
def getMnust10(location):
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
        plt.imshow(imgs[0],cmap=plt.get_cmap('gray'))
        

        #while (byte:=f.read(1)):
            


# %% Declarations
import network

nn_architecture = [
     {"input_dim": 2, "output_dim": 4, "activation": "relu"},
     {"input_dim": 4, "output_dim": 6, "activation": "relu"},
     {"input_dim": 6, "output_dim": 6, "activation": "relu"},
     {"input_dim": 6, "output_dim": 4, "activation": "relu"},
     {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
     ]

# %% Importing data

print("Importing Data...")




# %% Running Network

exit 
neto = network(nn_architecture
