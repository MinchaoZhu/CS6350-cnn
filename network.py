import numpy as np
import pickle
import sys
import time
from conv_layer import ConvolutionLayer
from fully_connected_layer import FullyConnectedLayer
from maxpooling_layer import MaxPoolingLayer
from softmax import Softmax
from relu import ReLu
from loss import  cross_entropy
from flatten import  Flatten


class Net:
    def __init__(self):

        lr = 0.01
        self.layers = []
        self.layers.append(
            ConvolutionLayer(inputs_channel=1, num_filters=6, width=5, height=5, padding=2, stride=1, learning_rate=lr,
                          name='conv1'))
        self.layers.append(ReLu())
        self.layers.append(MaxPoolingLayer(width=2, height=2, stride=2, name='maxpool2'))
        
        self.layers.append(
            ConvolutionLayer(inputs_channel=6, num_filters=16, width=5, height=5, padding=0, stride=1, learning_rate=lr,
                          name='conv3'))
        self.layers.append(ReLu())
        self.layers.append(MaxPoolingLayer(width=2, height=2, stride=2, name='maxpool4'))
        
        self.layers.append(
             ConvolutionLayer(inputs_channel=16, num_filters=120, width=5, height=5, padding=0, stride=1, learning_rate=lr,
                           name='conv5'))
    
        self.layers.append(ReLu())
        self.layers.append(Flatten())
        self.layers.append(FullyConnectedLayer(num_inputs=120, num_outputs=60, learning_rate=lr, name='fc6'))
        self.layers.append(ReLu())
        self.layers.append(FullyConnectedLayer
                           (num_inputs=60, num_outputs=2, learning_rate=lr, name='fc7'))
        self.layers.append(Softmax())
        self.lay_num = len(self.layers)

    def train(self, training_data, training_label, batch_size, epoch, weights_file):
        total_acc = 0
        for e in range(epoch):
            for batch_index in range(0, training_data.shape[0], batch_size):
                # batch input
                if batch_index + batch_size < training_data.shape[0]:
                    data = training_data[batch_index:batch_index + batch_size]
                    label = training_label[batch_index:batch_index + batch_size]
                else:
                    data = training_data[batch_index:training_data.shape[0]]
                    label = training_label[batch_index:training_label.shape[0]]
                loss = 0
                acc = 0
                start_time = time.time()
                
                for b in range(len(data)):
                    x = data[b]
                    y = label[b]
                   # print(y)
                    # forward pass
                    for l in range(self.lay_num):
                        output = self.layers[l].forward(x)
                        x = output
                    loss += cross_entropy(output, y)
                    if np.argmax(output) == np.argmax(y):
                        acc += 1
                        total_acc += 1
                    # backward pass
                    dy = y
                    for l in range(self.lay_num - 1, -1, -1):
                        dout = self.layers[l].backward(dy)
                        dy = dout
                # time
                end_time = time.time()
                batch_time = end_time - start_time
                remain_time = (training_data.shape[0] * epoch - batch_index - training_data.shape[
                    0] * e) / batch_size * batch_time
                hrs = int(remain_time) / 3600
                mins = int((remain_time / 60 - hrs * 60))
                secs = int(remain_time - mins * 60 - hrs * 3600)
                # result
                loss /= batch_size
                batch_acc = float(acc) / float(batch_size)
                training_acc = float(total_acc) / float((batch_index + batch_size) * (e + 1))
                print(
                    '=== Epoch: {0:d}/{1:d} === Iter:{2:d} === Loss: {3:.2f} === BAcc: {4:.2f} === TAcc: {5:.2f} === Remain: {6:d} Hrs {7:d} Mins {8:d} Secs ==='.format(
                        e, epoch, batch_index + batch_size, loss, batch_acc, training_acc, int(hrs), int(mins),
                        int(secs)))
        # dump weights and bias
        obj = []
        for i in range(self.lay_num):
            cache = self.layers[i].extract()
            obj.append(cache)
        with open(weights_file, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def test(self, data, label, test_size):
        toolbar_width = 40
        sys.stdout.write("[%s]" % (" " * (toolbar_width - 1)))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width))
        step = float(test_size) / float(toolbar_width)
        st = 1
        total_acc = 0
        for i in range(test_size):
            if i == round(step):
                step += float(test_size) / float(toolbar_width)
                st += 1
                sys.stdout.write(".")
                # sys.stdout.write("%s]a"%(" "*(toolbar_width-st)))
                # sys.stdout.write("\b" * (toolbar_width-st+2))
                sys.stdout.flush()
            x = data[i]
            y = label[i]
            #print(y)
            for l in range(self.lay_num):
                output = self.layers[l].forward(x)
                x = output
            if np.argmax(output) == np.argmax(y):
                total_acc += 1
        sys.stdout.write("\n")
        print('=== Test Size:{0:d} === Test Acc:{1:.2f} ==='.format(test_size, float(total_acc) / float(test_size)))

   