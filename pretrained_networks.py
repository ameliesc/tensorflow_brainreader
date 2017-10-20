import tensorflow as tf
from scipy.io import loadmat
from collections import OrderedDict

class Vgg19(object):

    def __init__(self, file_path = 'imagenet-vgg-verydeep-19.mat'):
        self.network_params = loadmat(file_path)

    def build(self):

        print("Building model...")
        self.network_layers = OrderedDict()
        for i in xrange(self.network_params['layers'].shape[1]):
            struct = self.network_params['layers'][0, i][
                                 0, 0]

            layer_type = struct[1][0]
            layer_name = str(struct[0][0])
        
            assert isinstance(layer_type, basestring)
            if layer_type == 'conv':
                w = struct[2][0, 0] 
                b = struct[2][0, 1][:, 0]
                if layer_name.startswith('fc'):
                    layer = FcLayer(w,b, name = layer_name)
                        
                elif layer_name.startswith('conv'):
                    strides = struct[4][0].tolist()
                    padding = "SAME"
                    layer = ConvLayer(w, b, strides = strides, padding = padding, name =  layer_name)
                else:
                    bad_value(layer_name)
                
            elif layer_type in ('relu', 'softmax'):
                    layer = Nonlinearity(layer_type, name = layer_name)
            elif layer_type == 'pool':
                    pooling_mode = str(struct[2][0])
                    layer  = Pooler(region=struct[3][0].tolist(), stride=struct[4][0].tolist(), mode=pooling_mode, name = layer_name)#not not returning indexes here!
                    
            else:
                raise Exception(
                    "Don't know about this '%s' layer type." % layer_type)
                


            self.network_layers[layer_name] = layer
        print("Done.")
        
    def compile(self,rgb):
        """
        load vgg parameters from file
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :output : dictionary containing each layer activation 
        """
        print("Compiling model...")
        # VGG_MEAN = [103.939, 116.779, 123.68]

        # rgb_scaled = rgb * 255.0# Convert RGB to BGR
        # red, green, blue = tf.split(rgb_scaled,3,3)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        # bgr = tf.concat([
        #     blue - VGG_MEAN[0],
        #     green - VGG_MEAN[1],
        #     red - VGG_MEAN[2],
        # ],3)
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        #### above needs fixing
        self.n_layers = len(self.network_layers)
        if isinstance(self.network_layers, (list, tuple)):
            layers = OrderedDict(zip(enumerate(self.network_layers)))
        else:
            assert isinstance(self.network_layers, OrderedDict), "Layers must be presented as a list, tuple, or OrderedDict"
            
        self.layers = self.network_layers
        x = rgb
        named_activations = OrderedDict()
        for name, layer in self.layers.iteritems():
            print('%s input shape: '% (name))
            print(x)
            print(layer)
            x = layer(x)
            layer_x = x
            named_activations[name] = layer_x
            
        print("Done.")
        return named_activations
        


class ConvLayer(object):

    def __init__(self, w, b ,strides, padding = "SAME", name = "conv"):
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
        x: [batch, in_height, in_width, in_channels] -> for later change greyscale ito 3d because this is how the network works tralalal
        """
        bias = tf.nn.bias_add(tf.nn.conv2d(input = x, filter = self.w  ,strides = self.strides, padding = self.padding, name=self.name), self.b)
        return tf.nn.relu(bias)#excluded biasterm for testing

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
        pool = tf.nn.max_pool_with_argmax(x,self.region, self.stride, padding = 'SAME', name = self.name)[0]
        # for now leaving out average since no average pool in vgg19
        return pool ##used argmax to work with unpooling layerm also adapt padding so its an input variable 

class FcLayer(object):
    def __init__(self, w, b, name):
        self.w = tf.constant(w, shape = w.shape)
        self.b = tf.constant(b, shape = b.shape)
        self.name = name
                                
    def __call__(self, x):
       # with tf.variable_scope(name) as scope:
        shape =  x.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
            x = tf.reshape(x, [-1, dim])
            
        weights = self.w
        biases = self.b

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
        return fc
                     
