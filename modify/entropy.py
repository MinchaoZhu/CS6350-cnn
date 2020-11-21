# loss
def cross_entropy(inputs, labels):

    # out_num = labels.shape[0]
    # p = np.sum(labels.reshape(1,out_num)*inputs)
    # loss = -np.log(p)
    return -np.log(np.sum(labels.reshape(1,labels.shape[0])*inputs))