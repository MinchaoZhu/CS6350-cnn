
import numpy as np

class ConvolutionLayer:
  # A Convolution layer

    # generate random matrix
    def generateRandomMatrix(self): 
        rand = self.channel * self.width * self.height;
        matrix = np.zeros((self.channel, self.width, self.height))
        for i in range (self.channel):
            for j in range (self.width):
                for k in range (self.height):
                    matrix[i][j][k] = np.random.normal(loc = 0, scale = np.sqrt(1. / rand))
        return matrix

    # create filter matrix
    def createFilterMatrix(self):
        for i in range(self.num_filters):
            # initialize filter matrix with random
            self.weights[i] = self.generateRandomMatrix()
        return 

    # initialization function to create convolutional layer instance
    def __init__(self, num_filters, inputs_channel, width, height, stride, padding, learning_rate):

        # number of convolutional filter as defined, it is required to be at least 1
        if num_filters <=0: 
             print("invalid filter number is provided.")
             return
        else:
            self.num_filters = num_filters
            # bias matrix is set to all zero initially
            self.bias = np.zeros((self.num_filters,1))
        # number of image channels. In the project, all image channel is decided to be 1 for gray image
        if inputs_channel <=0: 
             print("invalid channel number is provided.")
             return
        else:
            self.channel = inputs_channel
        # width of input images
        if width <=0: 
             print("invalid width of matrix is provided.")
             return
        else:
            self.width = width
        # height of input images
        if height <=0: 
             print("invalid height of matrix is provided.")
             return
        else:
            self.height = height
            # create random convolutional matrix
            self.weights = np.zeros((self.num_filters, self.channel, self.width, self.height))
        # step size of convolutional matrix's sliding step
        if stride <=0: 
             print("Value of stride needs to be greater than 0 to be meaningful.")
             return
        else:
            self.stride = stride
        # padding for result
        if padding < 0: 
             print("padding can't be negative value.")
             return
        else:
            self.padding = padding
        # learning rate for gradient descent study
        if learning_rate  < 0: 
            print("Please provide a positive learning rate")
        else:
            self.lr = learning_rate

    
    def paddingInclusion(self, inputs):
        # get input image's shape and calculate result image shape including padding size
        w = inputs.shape[0]
        h = inputs.shape[1]
        # new shape of matrix after considering padding
        new_w = 2 * self.padding + w
        new_h = 2 * self.padding + h
        out = np.zeros((new_w, new_h))
        # fill those zero spots by input values and rest will be left as it is
        for i in range(new_w):
            for j in range(new_h):
                if i >= self.padding and i < self.padding + w and j >= self.padding and j < self.padding + h:
                    out[i][j] = inputs[i-self.padding][j-self.padding]
        return out

    def sumResultOfMatrix(self, changedMatrix):
        f = changedMatrix.shape[0]
        w = changedMatrix.shape[1]
        h = changedMatrix.shape[2]
        result = 0
        for i in range(f):
            for j in range(w):
                for k in range(h):
                    result += changedMatrix[i][j][k]
        return result


    def calculateForwardMatrix(self, new_width, new_height):
        # corner check
        if new_width > 0 and new_height > 0:
            forward_matrix = np.zeros((self.num_filters, new_width, new_height))
            for f in range(self.num_filters):
                for w in range(new_width):
                    for h in range(new_height):
                        changedMatrix = self.inputs[:,w:w+self.width,h:h+self.height]*self.weights[f,:,:,:]
                        # sum up result to form forward matrix
                        forward_matrix[f,w,h]= self.sumResultOfMatrix(changedMatrix)+self.bias[f]
            return forward_matrix
        else:
            print("either width or height is invalid")
            return

    def forward(self, inputs):
        # get input shape, height, width and channel
        channel = inputs.shape[0]
        padding = 2 * self.padding
        stride = self.stride
        process_width = padding + inputs.shape[1]
        process_height = padding + inputs.shape[2]
        self.inputs = np.zeros((channel, process_width, process_height))
        for ch in range(channel):
            # process padding
            self.inputs[ch] = self.paddingInclusion(inputs[ch])
        new_width = (process_width - self.width)//stride + 1
        new_height = (process_height - self.height)//stride + 1
        return self.calculateForwardMatrix(new_width, new_height)

    def backward(self, dy):
        inputs = self.inputs
        weights = self.weights
        bias = self.bias
        src = np.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2]))
        wei = np.zeros((weights.shape[0], weights.shape[1], weights.shape[2]))
        bi = np.zeros((bias.shape[0], bias.shape[1], bias.shape[2]))

        fil, width, height = dy.shape
        if fil > 0 and width > 0 and height > 0:
            for f in range(fil):
                for w in range(width):
                    for h in range(height):
                        wei[f]+=dy[f,w,h]*self.inputs[:,w:w+self.width,h:h+self.height]
                        src[:,w:w+self.width,h:h+self.height]+=dy[f,w,h]*self.weights[f,:,:,:]
        else:
            return

        for f in range(fil):
            sum = 0;
            target_matrix = dy[f]
            for i in range (target_matrix.shape[0]):
                for j in range (target_matrix.shape[1]):
                    sum += target_matrix[i][j]
            bi[f] = sum

        self.weights -= self.lr * wei
        self.bias -= self.lr * bi
        return src
        