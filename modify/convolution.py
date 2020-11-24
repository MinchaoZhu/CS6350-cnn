
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

    
    def zero_padding(self, inputs):
        # get input image's shape and calculate result image shape including padding size
        w, h = inputs.shape[0], inputs.shape[1]
        # new shape of matrix after considering padding
        new_w = 2 * self.padding + w
        new_h = 2 * self.padding + h
        out = np.zeros((new_w, new_h))
        # fill those zero spots by input values and rest will be left as it is
        if new_w > 0 and new_h > 0: 
            out[self.padding:w+self.padding, self.padding:h+self.padding] = inputs
        return out

    def calculateForwardMatrix(self, new_width, new_height):
        # corner check
        if new_width > 0 and new_height > 0:
            forward_matrix = np.zeros((self.num_filters, new_width, new_height))
            for f in range(self.num_filters):
                for w in range(new_width):
                    for h in range(new_height):
                        # sum up result to form forward matrix
                        forward_matrix[f,w,h]=np.sum(self.inputs[:,w:w+self.width,h:h+self.height]*self.weights[f,:,:,:])+self.bias[f]
            return forward_matrix
        else:
            print("either width or height is invalid")
            return

    def forward(self, inputs):
        # get input shape, height, width and channel
        channel = inputs.shape[0]
        width = inputs.shape[1]+2*self.padding
        height = inputs.shape[2]+2*self.padding
        self.inputs = np.zeros((channel, width, height))
        for ch in range(inputs.shape[0]):
            # process padding
            self.inputs[ch] = self.zero_padding(inputs[ch])
        new_width = (width - self.width)//self.stride + 1
        new_height = (height - self.height)//self.stride + 1
        return self.calculateForwardMatrix(new_width, new_height)

    def backward(self, dy):

        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)

        fil, width, height = dy.shape
        if fil > 0 and width > 0 and height > 0:
            for f in range(fil):
                for w in range(width):
                    for h in range(height):
                        dw[f]+=dy[f,w,h]*self.inputs[:,w:w+self.width,h:h+self.height]
                        dx[:,w:w+self.width,h:h+self.height]+=dy[f,w,h]*self.weights[f,:,:,:]
        else:
            return

        for f in range(fil):
            db[f] = np.sum(dy[f, :, :])

        self.weights -= self.lr * dw
        self.bias -= self.lr * db
        return dx
        