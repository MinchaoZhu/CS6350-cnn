class ReLu:
    def forward(self, inputs):
        self.inputs = inputs
        relu = inputs.copy()
        # change the value that is less than 0 to 0
        relu[relu < 0] = 0
        return relu

    def backward(self, dy):
        inputs = dy.copy()
        inputs[self.inputs < 0] = 0
        return inputs