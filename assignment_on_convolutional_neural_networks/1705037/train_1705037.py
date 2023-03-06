
from abc import ABC, abstractmethod
import numpy as np
import math
import cv2
import pandas as pd
from sklearn.metrics import log_loss,f1_score,confusion_matrix,accuracy_score
import csv
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import pickle
from tqdm import tqdm


'''
    Model Architecture
'''
### Model Architecture Start ###

class Layer(ABC):

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    @abstractmethod
    def clear_params(self):
        pass


'''
    Convolution
'''
class Convolution(Layer):
    # nc = number of filters/ num of output channels
    # f = filter dimension
    # s = stride
    # p = padding

    def __init__(self, nc, f, s, p, learning_rate=0.00001):
        self.nc = nc
        self.f = f
        self.s = s
        self.p = p
        self.w = None
        self.b = np.zeros(nc)
        self.learning_rate = learning_rate

    #for vectorized implementation : https://blog.ca.meron.dev/Vectorized-CNN/
    def get_strided_windows(self,input, output_size, kernel_size, padding=0, stride=1,dilation=0):
        input_array = np.copy(input)

        if dilation :
            input_array = np.insert(input_array, range(1, input.shape[2]), 0, axis=2)
            input_array = np.insert(input_array, range(1, input.shape[3]), 0, axis=3)

        if padding :
            input_array = np.pad(input_array, pad_width=((0,), (0,), (padding,), (padding,)), mode='constant', constant_values=(0.,))

        batch_size, input_channels, output_height, output_width = output_size
        output_batch_size = input.shape[0]
        output_channels = input.shape[1]

        batch_str, channel_str, kern_h_str, kern_w_str = input_array.strides

        return np.lib.stride_tricks.as_strided(
            input_array,
            (output_batch_size, output_channels, output_height, output_width, kernel_size, kernel_size),
            (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str)
        )

    def forward(self, data):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters.
        """

        n, c, h, w = data.shape
        out_h = (h - self.f + 2 * self.p) // self.s + 1
        out_w = (w - self.f + 2 * self.p) // self.s + 1

        output_size = n, c, out_h, out_w

        windows = self.get_strided_windows(data, output_size, self.f, self.p, self.s)

        # xavier initialization of weights
        if self.w is None:
            self.w = np.random.randn(self.nc, c, self.f, self.f) * np.sqrt(2.0 / (self.f * self.f * c))
            self.b = np.zeros(self.nc)

        out = np.einsum('bihwkl,oikl->bohw', windows, self.w) + self.b[None, :, None, None]
        if np.inf in out:
            print("out",out)

        self.input = data
        self.windows = windows

        return out

    def backward(self, delta):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: dx, dw, and db relative to this module
        """

        if self.p == 0:
            padding = self.f - 1
        else:
            padding = self.p

        delta_strides = self.get_strided_windows(delta, self.input.shape, self.f, padding=padding, stride=1, dilation=self.s - 1)
        rot_kern = np.rot90(self.w, 2, axes=(2, 3))

        db = np.sum(delta, axis=(0, 2, 3))
        dw = np.einsum('bihwkl,bohw->oikl', self.windows, delta)
        dx = np.einsum('bohwkl,oikl->bihw', delta_strides, rot_kern)
        
        #normalizing over entries in batch
        # dw /= self.input.shape[0]
        # db /= self.input.shape[0]

        self.w -= self.learning_rate*dw
        self.b -= self.learning_rate*db

        return dx
    
    def clear_params(self):
        self.input = None
        self.windows = None

'''
    Max Pooling
'''
class MaxPool(Layer):

    def __init__(self, s, f):
        super().__init__()
        self.s = s
        self.f = f

    def forward(self, image):
        self.image = image
        m, c, h, w = image.shape
        width_out = math.floor((w - self.f) / self.s) + 1
        height_out = math.floor((h - self.f) / self.s) + 1
        # output = np.zeros([m, c, height_out, width_out])
        # for i in range(height_out):
        #     for j in range(width_out):
        #         output[:, :, i, j] = np.max(image[:, :, i * self.s:i * self.s + self.f,
        #                                  j * self.s:j * self.s + self.f], axis=(2, 3))
        # return output

        batch_stride, channel_stride, height_stride, width_stride = image.strides
        input_windows = np.lib.stride_tricks.as_strided(image,
            shape = (m, c, height_out, width_out, self.f, self.f),
            strides = (batch_stride, channel_stride, height_stride * self.s, width_stride * self.s, height_stride, width_stride)
        )
        output = np.max(input_windows, axis=(4, 5))

        if self.s == self.f:
            mask = output.repeat(self.s,axis=-2).repeat(self.s,axis=-1)
            h_pad = h - mask.shape[-2]
            w_pad = w - mask.shape[-1]
            mask = np.pad(mask,((0,0),(0,0),(0,h_pad),(0,w_pad)),'constant')
            mask = np.equal(image, mask)
            self.cache = mask
        return output

    def backward(self,output):
        n, c, h, w = self.image.shape
        _, _, height_out, width_out = output.shape
        if self.s == self.f:
            dx = output.repeat(self.s,axis=-2).repeat(self.s,axis=-1)
            mask = self.cache
            h_pad = h - dx.shape[-2]
            w_pad = w - dx.shape[-1]
            dx = np.pad(dx,((0,0),(0,0),(0,h_pad),(0,w_pad)),'constant')
            dx = np.multiply(dx,mask)
            return dx
        else:
            dx = np.zeros(self.image.shape)
            # print("output",output)
            for i in range(height_out):
                for j in range(width_out):
                    x_masked = self.image[:, :, i * self.s:i * self.s + self.f,
                                    j * self.s:j * self.s + self.f]
                    max_x_masked = np.max(x_masked, axis=(2, 3), keepdims=True)
                    dx_masked = (x_masked == max_x_masked)
                    dx[:, :, i * self.s:i * self.s + self.f,
                    j * self.s:j * self.s + self.f] += dx_masked * output[:, :, i, j][:, :, None, None]
            return dx
    
    def clear_params(self):
        self.image = None
        self.cache = None


'''
    Fully Connected Layer
'''
class FullyConnected(Layer):

    def __init__(self, output_size, learning_rate=0.00001):
        self.output_size = int(output_size)
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = np.zeros(output_size)

    def forward(self, input):
        self.input = np.copy(input)
        input_size = input.shape[0]
        if self.weights is None:
            self.weights = np.random.randn(input.shape[1],self.output_size) * np.sqrt(2.0 / input.shape[1])
        return np.dot(input,self.weights) + self.bias

    def backward(self, delta):
        output = np.dot(delta,self.weights.T)
        # dw = np.dot(delta, self.input.T)/delta.shape[1]
        dw = np.dot(self.input.T,delta)
        db = np.sum(delta, axis=0)

        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

        return output
    
    def clear_params(self):
        self.input = None


'''
    ReLU Activation
'''
class ReLu(Layer):
    def __init__(self):
        self.output = None
        self.delta = None

    def forward(self, input):
        self.image = input
        self.output = np.copy(input)
        self.output[self.output < 0] = 0
        return self.output

    def backward(self, output):
        image = np.copy(self.image)
        image[image > 0] = 1
        image[image < 0] = 0
        output = output * image
        return output
    
    def clear_params(self):
        self.output = None
        self.delta = None
        self.image = None

'''
    Softmax Activation
'''
class Softmax(Layer):
    def __init__(self):
        self.output = None
        self.delta = None
        self.y = None

    def forward(self, input):
        v = np.exp(input-np.max(input,axis=1,keepdims=True))
        v = v / np.sum(v, axis=1, keepdims=True) 
        return v

    def backward(self, input):
        return input
    
    def clear_params(self):
        pass

'''
    Flatten Layer
'''
class Flattening(Layer):

    def forward(self,input):
        image = np.copy(input)
        self.image = image
        flattened_image = np.reshape(image, (image.shape[0], -1))
        return flattened_image

    def backward(self,input):
        return np.reshape(input, self.image.shape)
    
    def clear_params(self):
        self.image = None

### Model Architecture End ###

'''
    Model Class
'''
class Model:

    def __init__(self,alpha=0.00001):

        layers_list = []

        layers_list.append(Convolution(6, 5, 1, 0,alpha))
        layers_list.append(ReLu())
        layers_list.append(MaxPool(2, 2))


        layers_list.append(Convolution(16, 5, 1, 0,alpha))
        layers_list.append(ReLu())
        layers_list.append(MaxPool(2, 2))
        
        layers_list.append(Flattening())
        layers_list.append(FullyConnected(120,alpha))

        layers_list.append(ReLu())
        layers_list.append(FullyConnected(84,alpha))
        layers_list.append(ReLu())

        layers_list.append(FullyConnected(10,alpha))
        layers_list.append(Softmax())

        self.layers_list = layers_list


    def train(self,X,y):
        op = np.copy(X)
        layer_count=0
        for layer in self.layers_list:
            op = layer.forward(op)
            layer_count+=1

        y = np.eye(10)[y].astype(int)
        delta = (op - y)/y.shape[0]

        back_layer_count=0
        for layer in reversed(self.layers_list):
            delta = layer.backward(delta)
            back_layer_count+=1

        return y,op


    def predict(self, X):
        op = np.copy(X)
        for layer in self.layers_list:
            op = layer.forward(op)
        return op

    def save(self):
        #use pickle to save self object
        for layer in self.layers_list:
            layer.clear_params()
        filename = '1705037_model.pickle'
        filehandler = open(filename, 'wb') 
        pickle.dump(self, filehandler)


    def load(self):
        #use pickle to load self object
        filename = '1705037_model.pickle'
        filehandler = open(filename, 'rb')
        self = pickle.load(filehandler)

'''
    Data Loader Functions
'''

def get_images_from_directory(directory, size=(28, 28)):
    image_paths = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    images = []

    for path in image_paths:
        image = cv2.imread(directory + path)
        image = cv2.resize(image, size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (255.0-np.array(image)) / 255.0
        images.append(image)
    output = np.array(images)
    #training c gives error here
    output = np.transpose(output, (0, 3, 1, 2))
    return image_paths,output

def load_dataset(list_of_training_set, size=(28, 28), number_of_images=-1):
    global kaggle
    list_of_y = []
    list_of_image_paths=[]

    for training_set in list_of_training_set:
        if kaggle == True:
            csv_path = '/kaggle/input/numta//'+training_set+'.csv'
        else:
            csv_path = 'NumtaDB//'+training_set+'.csv'
        df = pd.read_csv(csv_path)
        list_of_image_paths.append(training_set+'//'+df['filename'].values)
        list_of_y.append(df['digit'].values)
    
    image_paths = np.concatenate(list_of_image_paths)
    y = np.concatenate(list_of_y)

    #shuffle different training sets
    permutation = np.random.permutation(image_paths.shape[0])
    image_paths = image_paths[permutation]
    y = y[permutation]

    images = []
    if number_of_images==-1:
        number_of_images = len(image_paths)

    count = 0

    for path in image_paths:
        if kaggle == True:
            image = cv2.imread('/kaggle/input/numta/'+path)
        else :
            image = cv2.imread('NumtaDB/'+path)
        image = cv2.resize(image,size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (255.0-np.array(image)) / 255.0
        # print(image)
        images.append(image)

        count+=1
        if count==number_of_images:
            break
    
    output = np.array(images)
    output = np.transpose(output, (0, 3, 1, 2))
    return output,np.array(y[:number_of_images])

def train_test_split(X,y,split=0.8):
    X_train = X[:int(X.shape[0]*split),:,:,:]
    y_train = y[:int(y.shape[0]*split)]
    X_valid = X[int(X.shape[0]*split):,:,:,:]
    y_valid = y[int(y.shape[0]*split):]
    return X_train,y_train,X_valid,y_valid

''''
    Training Function
'''
def train_model_and_report(model,X,y,batch_size,epoch,learning_rate,classes):
    X_train,y_train,X_valid,y_valid = train_test_split(X,y,split=0.8)
    file = open(stats_directory+'/stats_'+str(alpha)+'.csv', 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(["Epoch","Training Loss","Validation Loss","Validation Accuracy","Macro-f1"])

    best_f1_score = 0
    best_conf = None

    for e in tqdm(range(epoch)):
        print("\n\nEpoch",e)
        print("========================================")
        print("Training")
        loss=0
        acc=0
        f1=0
        batch_count=0
        row = [e]

        for i in range(0,X_train.shape[0],batch_size):
            # print("\tbatch",i//batch_size)

            y_out, y_pred = model.train(X_train[i:i+batch_size],y_train[i:i+batch_size])
            # print("y_out",y_out)
            y_true = np.zeros((y_out.shape[0],classes))
            
            for j in range(y_out.shape[0]):
                y_true[j, y_out[j]] = 1  # generating one-hot encoding of y_train
            loss += log_loss(y_true, y_pred)
            acc += accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
            f1 += f1_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average='macro')
            batch_count+=1
        print("\tloss",loss/batch_count)
        row.append(loss/batch_count)
        print("\taccuracy",acc/batch_count)
        print("\tf1_score",f1/batch_count)
        
        #validation
        print("Validation")
        y_true = np.zeros((y_valid.shape[0],classes))
        y_pred = model.predict(X_valid)
        for j in range(y_valid.shape[0]):
            y_true[j, y_valid[j]] = 1

        val_loss = log_loss(y_true, y_pred)
        print("\tloss",val_loss)
        row.append(val_loss)

        val_acc = accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
        print("\taccuracy",val_acc)
        row.append(val_acc)

        f1 = f1_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average='macro')
        print("\tf1_score",f1)
        row.append(f1)
        
        conf_matrix = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
        if f1>=best_f1_score:
            best_f1_score = f1
            best_conf = conf_matrix
            try:
                model.save()
            except:
                print("Error in saving model")


        writer.writerow(row)
        # print("\tconfusion_matrix\n",confusion_matrix(np.argmax(y_true, axis=0), np.argmax(y_pred, axis=0)))
        if e==epoch-1:
            print("final confusion matrix")
            print(best_conf)
            #plot confusion matrix
            plt.figure(figsize=(10,10))
            
            sns.heatmap(best_conf, annot=True, cmap='Blues', fmt='g')

            # Add labels and title
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix Heatmap For Learning Rate '+str(learning_rate))
            # plt.show()
            plt.savefig(plots_directory+'/confusion_matrix_'+str(alpha)+'.png')

if __name__ == "__main__":
    # Statistics generation
    stats_directory = "results/stats"
    if not os.path.exists(stats_directory):
        os.makedirs(stats_directory)

    plots_directory = "results/plots"
    if not os.path.exists(plots_directory):
        os.makedirs(plots_directory)

    # Hyperparameters
    batch_size = 32
    epoch = 100

    
    try:
        alpha = float(sys.argv[1])
    except:
        alpha = 0.001
    
    try:
        num_images = int(sys.argv[2])
    except:
        num_images = 1000
    
    try:
        kaggle = int(sys.argv[3])
    except:
        kaggle = 0

    classes = 10

    #load_dataset call
    list_of_training_sets = ['training-a','training-b','training-c']
    X,y = load_dataset(list_of_training_sets,(28,28),num_images)

    model = Model(alpha)
    try :
        train_model_and_report(model,X,y,batch_size,epoch,alpha,classes)
    except:
        traceback.print_exc()
        exit(0)