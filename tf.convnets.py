import tensorflow as tf


def maxpool_positions(relu_im, stride = (2, 2), region = (2, 2)):  ## this function needs to be translated into tensorflow!!!!!  probs doesnt work so how we gotta do thisss? 
    pooled_im = max_pool_2d(relu_im, ds = region, st = stride)
    positions = tt.grad(pooled_im.sum(), relu_im)
    return positions


class ConvLayer(object):

    def __init__(self, w, b ,strides, padding = "VALID", name = "conv":
                 """
                 w is the kernel, an ndarray of shape (n_rows, n_cols, n_in_maps, n_out_maps) according to tf conventions
                 b is the bias, an ndarray of shape (n_output_maps, )
                 border_mode: default is VALID {"VALID, SAME"} see https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
                 """

                 self.w = tf.Variable(tf.constant(w, shape = w.shape))
                 self.b = tf.Variable(tf.constant(b, shape = b.shape))
                 self.strides = tf.constant(strides)
                 self.padding = padding
                 self.name = name

    def __call__(self,x):
                 """
                 x: [batch, in_height, in_width, in_channels] -> for later change greyscale into 3d because this is how the network works tralalal
                 """
                 return tf.nn.bias_add(tf.tf.nn.conv2d(input = x, filters = self.w  ,strides = self.strides, padding = self.padding, use_cudnn_on_gpu=None, data_format=None, name=self.name), self.b)

class Nonlinearity(object):
    def __init__(self, activation, name):
    """
    activation: a name for the activation function. {'relu', 'sig, 'tanh', 'softmax' ...}
    """
        self.activation = activation
        self.name = name

    def __call__(self, x):
        if self.activation  in 'relu':
            return tf.nn.relu_layer(x, self.w, self.b, name = self.activation)

        elif self.activation  in 'softmax':
            return tf.nn.softmax(x, name = self.name)

class Pooler(object):

    def __init__(self, region, stride = None, mode = 'max' , name):

        assert len(region) == 2, 'Region must consist of two integers.  Got: %s' % (region, )
        if stride is None:
            stride = region
        assert len(stride) == 2, 'Stride must consist of two integers.  Got: %s' % (stride, )
        self.region = [1] + region + [1] # so it can be fed into tensorflow max_pool who wants a list of length 4
        self.stride = [1] + stride + [1]
        self.name = name

    def __call__(self,x):
        """
        param x: 4-D tensor  [batch, height, width, channels]
        returns: [batch, heigt/ds[0], width/ds[0], channels]
        """
        # for now leaving out average since no average pool in vgg19
        return tf.nn.max_pool(x,self.region, self.stride, pooling = 'SAME', name = self.name)

            
class Switches(object):
    
    def __init__(self, region, stride = None):
        assert len(region) == 2, 'Region must consist of two integers.  Got: %s' % (region, )
        if stride is None:
            stride = region
        assert len(stride) == 2, 'Stride must consist of two integers.  Got: %s' % (stride, )
        self.region = region
        self.stride = stride

    def max_pool_positions(self, x, stride, region):
        pooled_im = tf.nn.max_pool(x,self.region, self.stride, pooling = 'SAME')
        positions = tf.gradients(pooled_im)


    def __call__(self, x):
        return maxpool_positions(x, stride = self.stride, region = self.region)

class ConvNet(object): ## jsut copie pasted for now shuld be changed

    def __init__(self, layers):
        """
        :param layers: Either:
            A list of layers or
            An OrderedDict<layer_name: layer>
        """
        self.n_layers = len(layers)
        if isinstance(layers, (list, tuple)):
            layers = OrderedDict(zip(enumerate(layers)))
        else:
            assert isinstance(layers, OrderedDict), "Layers must be presented as a list, tuple, or OrderedDict"
        self.layers = layers

    def __call__(self, inp):
        """
        :param inp: An (n_samples, n_colours, size_y, size_x) input image
        :return: An (n_samples, n_feature_maps, map_size_y, map_size_x) feature representation.
        """
        return self.get_named_layer_activations(inp).values()[-1]

    def get_named_layer_activations(self, x):
        """
        :returns: An OrderedDict<layer_name/index, activation>
            If you instantiated the convnet with an OrderedDict, the keys will correspond to the keys for the layers.
            Otherwise, they will correspond to the index which identifies the order of the layer.
        """
        # named_activations = OrderedDict()
        # for name, layer in self.layers.iteritems():
        #     print '%s input shape: %s' % (name, x.ishape)
        #     x = layer(x)
        #     named_activations[name] = x
        # print '%s output shape: %s' % (name, x.ishape)
        # return named_activations
        named_activations = OrderedDict()
        for name, layer in self.layers.iteritems():
             print '%s input shape: %s' % (name, x.ishape)
             #print self.layers
             #print name
             if "switch" in name:
                 switch = layer(x)
                 layer_x = switch
             else:
                 x = layer(x)
                 layer_x = x
             named_activations[name] = layer_x
             print '%s output shape: %s' % (name, x.ishape)

        return named_activations
                 
                
