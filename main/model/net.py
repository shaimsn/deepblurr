"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions

    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_channels_L1 = params.num_channels_L1
        self.filter_size_L1 = params.filter_size_L1
        self.num_resblocks = params.num_resblocks # This number is set in params.json
        
        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        padding_L1 = int((self.filter_size_L1-1)/2)
        self.conv_in_L1 = nn.Conv2d(3, self.num_channels_L1, self.filter_size_L1, stride=1, padding=padding_L1)
        self.conv_RB_L1 = nn.Conv2d(self.num_channels_L1, self.num_channels_L1, self.filter_size_L1, stride=1,
                                    padding=padding_L1)
        self.conv_out_L1 = nn.Conv2d(self.num_channels_L1, 3, self.filter_size_L1, stride=1, padding=padding_L1)
        self.relu1 = nn.ReLU()

        # self.bn1 = nn.BatchNorm2d(self.num_channels)
        # self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        # self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        # self.fc1 = nn.Linear(8*8*self.num_channels*4, self.num_channels*4)
        # self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        # self.fc2 = nn.Linear(self.num_channels*4, 6)
        # self.dropout_rate = params.dropout_rate

    def res_block(self, s):
        """
        This function defines one ResBlock as defined by Nah et. al in:
        http://openaccess.thecvf.com/content_cvpr_2017/papers/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf
        :param s: the input to the network
        :return:
        """
        inp_copy = s
        s = self.conv_RB_L1(self.relu1(self.conv_RB_L1(s)))
        s = s + inp_copy  # skip connection
        return s

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 3 x 64 x 64 with the deblurred versions of each image.

        Note: the dimensions after each step are provided
        """
        #                                                  -> batch_size x 3 x 64 x 64
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = self.conv_in_L1(s)  # transforms image to feature block of 64
        for i in range(self.num_resblocks):
            s = self.res_block(s)  # iterate on resblocks
        L1 = self.conv_out_L1(s)  # transforms back to image

        # s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 64 x 64
        # s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 32 x 32
        # s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 32 x 32
        # s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 16 x 16
        # s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 16 x 16
        # s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*4 x 8 x 8

        # flatten the output for each image
        # s = s.view(-1, 8*8*self.num_channels*4)             # batch_size x 8*8*num_channels*4

        # apply 2 fully connected layers with dropout
        # s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
        #     p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
        # s = self.fc2(s)                                     # batch_size x 6

        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        # return F.log_softmax(s, dim=
        return L1


def loss_fn(model_outputs, true_outputs):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x num_channels x width  x height
        inputs: (Variable) dimension batch_size x num_channels x width  x height

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    #

    num_examples, channels, width, height = model_outputs.size()
    assert(model_outputs.size() == true_outputs.size())

    # return -torch.sum(outputs[range(num_examples), labels])/num_examples
    # TODO: is this correct?
    norm_factor = 1./(num_examples*channels*width*height)
    # return norm_factor*torch.sum([torch.norm(model_outputs[i]-true_outputs[i] for i in num_examples)])
    return norm_factor*torch.sum(torch.norm(model_outputs-true_outputs))


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
