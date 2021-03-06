from collections import OrderedDict
import tensorflow as tf




class ConvLayer(object):

    def __init__(self, w, b ,strides, padding = "VALID", name = "conv"):
                 """
                 w is the kernel, an ndarray of shape (n_rows, n_cols, n_in_maps, n_out_maps) according to tf conventions
                 b is the bias, an ndarray of shape (n_output_maps, )
                 border_mode: default is VALID {"VALID, SAME"} see https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
                 """

                 self.w = tf.constant(w, shape = w.shape)
                 self.b = tf.constant(b, shape = b.shape)
                 self.strides = strides
                 self.padding = padding
                 self.name = name

    def __call__(self,x):
        """
        x: [batch, in_height, in_width, in_channels] -> for later change greyscale into 3d because this is how the network works tralalal
        """
        return tf.nn.conv2d(input = x, filter = self.w  ,strides = self.strides, padding = self.padding, name=self.name) #excluded biasterm for testing

class Nonlinearity(object):

    def __init__(self, activation, name = 'relu'):
        """   activation: a name for the activation function. {'relu', 'sig, 'tanh', 'softmax' ...}  """
        
        self.activation = activation
        self.name = name

    def __call__(self, x):
        if self.activation  in 'relu':
            return tf.nn.relu(x, name = self.name)

        elif self.activation  in 'softmax':
            return tf.nn.softmax(x, name = self.name)

class Pooler(object):

    def __init__(self, region, stride = None, mode = 'max' , name = 'pool'):
        
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
        return tf.nn.max_pool_with_argmax(x,self.region, self.stride, padding = 'SAME', name = self.name) ##used argmax to work with unpooling layerm also adapt padding so its an input variable 

            


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
             print '%s input shape: %s' % (name, tf.shape(x))
             #print self.layers
             #print name
             # if "indexes" in name:
             #     switch = layer(x)
             #     layer_x = switch
             # else:
             x = layer(x)
             layer_x = x
             named_activations[name] = layer_x 
             print '%s output shape: %s' % (name, tf.shape(x))

        return named_activations
                 
                
