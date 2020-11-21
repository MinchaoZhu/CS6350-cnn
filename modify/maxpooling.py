class MaxPoolingLayer:
    def __init__(self, width, height, stride):
        if width > 0:
            self.width = width
        else:
            sys.exit("width cannot be negative") 
        if height > 0:
            self.height = height
        else:
            sys.exit("height cannot be negative") 
        if stride > 0:
            self.stride = stride
        else:
            sys.exit("stride cannot be negative") 

    def new_demension(self, current, old, stride):
        if current > 0 and old > 0 and stride > 0:
            return (current - old) // stride + 1
        else:
            return 0

    def forward_out(self, c_len, new_width, new_height):
        out = 0
        if c_len >= 0 and new_width >= 0 and new_height >= 0:
            out = np.zeros((c_len, new_width, new_height))
        else:
            sys.exit("Inputs should not be negative") 
        for c in range(c_len):
            for w in range(new_width):
                for h in range(new_height):
                    #np.max get the max value
                    out[c, w, h] = np.max(
                        self.inputs[c, w * self.stride:w * self.stride + self.width, h * self.stride:h * self.stride + self.height])
        return out

    def forward(self, inputs):
        if inputs is not None:
            self.inputs = inputs
        else:
            sys.exit("Inputs should not be None") 

        c_len, w_len, h_len = inputs.shape
        new_width = self.new_demension(w_len, self.width, self.stride)
        new_height = self.new_demension(h_len, self.height, self.stride)
        result = self.forward_out(c_len,new_width, new_height)
        if result is not None:
            return result
        else:
            sys.exit("error: output is none") 

    def backward_out(self, c_len, w_len, h_len, delta_y):
        dx = np.zeros(self.inputs.shape)
        for c in range(c_len):
            for w in range(0, w_len, self.width):
                for h in range(0, h_len, self.height):
                    st = np.argmax(self.inputs[c, w:w + self.width, h:h + self.height])
                    (idx, idy) = np.unravel_index(st, (self.width, self.height))
                    dx[c, w + idx, h + idy] = delta_y[c, w // self.width, h // self.height]
        return dx

    def backward(self, delta_y):
        c_len, w_len, h_len = self.inputs.shape
        if c_len < 0 or w_len < 0 or h_len < 0:
            sys.exit("Inputs should not be negative")
        result = self.backward_out(c_len, w_len, h_len, delta_y)
        # for c in range(c_len):
        #     for w in range(0, w_len, self.width):
        #         for h in range(0, h_len, self.height):
        #             #np.argmax get the index of max value , and the index is the index after flattening the array
        #             st = np.argmax(self.inputs[c, w:w + self.width, h:h + self.height])
        #             # get the original index of max value from the flattened index
        #             (idx, idy) = np.unravel_index(st, (self.width, self.height))
        #             dx[c, w + idx, h + idy] = dy[c, w // self.width, h // self.height]
        return result

    def extract(self):
        return