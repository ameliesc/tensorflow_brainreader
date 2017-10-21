import data_dict
import tensorflow as tf
import numpy as np


class Vgg19_deconv(argmax_dict, data_path = None):
    """
    Vgg19 deconvolution network that deconvolves from called layer onwards- pretty amazing eh?
    :param argmax_dict: dictionary containing the pooling argmax - can be obtained from Vgg19
    :pram data_path: optional,full path to where the vgg19.mat file is
    each layer output can be accesessed with self.layer_activations['layername']
    """
    def __init__(self, argmax_dict, data_path = None):
        self.argmax1 = argmax_dict['pool1']
        self.argmax2 = argmax_dict['pool2']
        self.argmax3 = argmax_dict['pool3']
        self.argmax4 = argmax_dict['pool4']
        self.argmax5 = argmax_dict['pool5']
        if data_path is None:
            self.data_dict = data_dict()
        else:
            self.data_dict = data_dict(data_path)
        print("npy file loaded")

    def build(self):
        start_time = time.time()
        print("Building model...")

        prob = tf.nn.softmax(fc8, name="prob")
        fc8 = Fc_layer( "fc8")
        fc7 = Fc_layer("fc7")
        fc6 = Fc_layer("fc6")
        unpool5= Unpool_layer(self.argmax5, 'unpool5')
        deconv5_4 = Deconv_layer(self.get_conv_filter("deconv5_4"),self.get_bias("deconv5_4"), "deconv5_4")
        deconv5_3 = Deconv_layer(self.get_conv_filter("deconv5_3"),self.get_bias("deconv5_3"), "deconv5_3")
        deconv5_2 = Deconv_layerself.get_conv_filter("deconv5_2"),self.get_bias("deconv5_2"), "deconv5_2")
        deconv5_1 = Deconv_layer(self.get_conv_filter("deconv5_1"),self.get_bias("deconv5_1"), "deconv5_1")
        unpool4 = Unpool_layer(self.argmax4, 'unpool4')
        deconv4_4 = Deconv_layer(self.get_conv_filter("deconv4_4"),self.get_bias("deconv4_4"), "deconv4_4")
        deconv4_3 = Deconv_layer(self.get_conv_filter("deconv4_3"),self.get_bias("deconv4_3"), "deconv4_3")
        deconv4_2 = Deconv_layer(self.get_conv_filter("deconv4_2"),self.get_bias("deconv4_2"), "deconv4_2")
        deconv4_1 = Deconv_layer(self.get_conv_filter("deconv4_1"),self.get_bias("deconv4_1"), "deconv4_1")
        unpool3 = Unpool_layer(self.argmax3, 'unpool3')
        deconv3_4 = Deconv_layer(self.get_conv_filter("deconv3_4"),self.get_bias("deconv3_4"), "deconv3_4")
        deconv3_3 = Deconv_layer(self.get_conv_filter("deconv3_3"),self.get_bias("deconv3_3"), "deconv3_3")
        deconv3_2 = Deconv_layer(self.get_conv_filter("deconv3_2"),self.get_bias("deconv3_2"), "deconv3_2")
        deconv3_1 = Deconv_layer(self.get_conv_filter("deconv3_1"),self.get_bias("deconv3_1"), "deconv3_1")

        unpool2 = Unpool_layer(self.argmax2,'unpool2')
        deconv2_2 = Deconv_layer(self.get_conv_filter("deconv2_2"),self.get_bias("deconv2_2"), "deconv2_2")
        deconv2_1 = Deconv_layer(self.get_conv_filter("deconv2_1"),self.get_bias("deconv2_1"), "deconv2_1")

        unpool1 = Unpool_layer(self.argmax1, 'unpool1')
        deconv1_2 = Deconv_layer(self.get_conv_filter("deconv1_2"),self.get_bias("deconv1_2"), "deconv1_2")
        deconv1_1 = Deconv_layer(self.get_conv_filter("deconv1_1"),self.get_bias("deconv1_1"),"deconv1_1"))
        self.data_dict = None

        layers_list = [prob, fc8, fc7, fc6, unpool5,deconv5_4, deconv5_3, deconv5_2,deconv5_1,unpool4,deconv4_4,deconv4_3,deconv4_2.deconv4_1,unpool3,deconv3_4,deconv3_3,deconv3_2,deconv3_1,unpool2,deconv2_2,deconv2_1,unpool1,deconv1_2,deconv1_1]

        layer_names =['prob', 'fc8',
        'relu7', 'fc7', 'relu6', 'fc6', 'pool5', '4_relu5', '4_conv5', '3_relu5', '3_conv5', '2_relu5', '2_conv5',
        '1_relu5', '1_conv5', 'pool4', '4_relu4', '4_conv4', '3_relu4', '3_conv4', '2_relu4', '2_conv4', '1_relu4',
        '1_conv4', 'pool3', '4_relu3', '4_conv3', '3_relu3', '3_conv3', '2_relu3', '2_conv3', '1_relu3', '1_conv3',
        'pool2', '2_relu2', '2_conv2', '1_relu2', '1_conv2', 'pool1', '2_relu1', '2_conv1', '1_relu1', '1_conv1']

        self.layer_activations = OrderedDict(zip(layer_names,layers_list))
        print("build model finished: %ds" % (time.time() - start_time))

        ###rewrite above, no need to enter inbetween variables right

    def compile(from_layer, features):
        """
        :param from_layer: string declaring the layer from which deconvolution is supposed to be started
        :param features: input features for corresponding layer
        :output : returns the output of the last layer of deconvnet
        """
        if from_layer.startswith('fc'):
            shape_weights = tf.shape(self.get_fc_weight(self, from_layer))
            shape_features = features.shape
            if  shape_weights == shape_features:
                pass
            else:
                raise Exception("input has dimension: %d should be %d", %(shape_features,shape_features))

        if from_layer.startswith('conv'):
            shape_weights = tf.shape(self.get_conv_filter(self, from_layer))
            shape_features = features.shape
            if  shape_weights == shape_features:
                pass
            else:
                raise Exception("input has dimension: %d should be %d", %(shape_features,shape_features))
            
        for name, layer in self.layer_activations.iteritems():
            if from_layer is not None:
                if from_layer != name:
                    continue
            else: 
                from_layer = None
            x = layer(x)
            if name == 'fc8':
                x = tf.nn.relu(x)
            elif name == 'fc7':
                x = tf.nn.relu(x)
            layer_x = x
            self.layer_activations[name] = layer_x
        return layer_x

            
    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

 class Fc_layer(self, bottom, name):

    def __init__(self,name):
         self.name = name

    def fc_layer(self,bottom,name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc
    def __call__(self,x):
        return self.fc_layer(x,self.name)

 class  Deconv_layer(object):

     def __init__(self, W,b name):
     self.W = W #self.get_conv_filter(name[2:])
     self.b = b # self.get_bias(name[2:])

     def __call__(self, x):
         """
         :param x: a 4D tensor - output from previous layer
         """
         x_shape = tf.shape(x)
         out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
     
         return tf.nn.conv2d_transpose(x, self.W, out_shape, [1, 1, 1, 1], padding=padding) + self.b

 class Unpool_layer(object):

    def __init__(self, raveled_argmax,name):
        self.name = name
        self.raveled_argmax = raveled_argmax
        
    def unravel_argmax(self, argmax, shape):
        output_list = []
        output_list.append(argmax // (shape[2] * shape[3]))
        output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
        return tf.stack(output_list)

    def unpool(self, x, raveled_argmax, name):
        if name = "unpool5":
            name = "conv5_4"
        elif name = "unpool4":
            name = "conv4_4"
        elif name = "unpool3":
            name = "conv3_4"
        elif name = "unpool2":
            name = "conv2_2"
        elif name = "unpool1":
            name = "conv1_2"
            
        out_shape = tf.shape(self.get_conv_filter(name))
        argmax = self.unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
        output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

        height = tf.shape(output)[0]
        width = tf.shape(output)[1]
        channels = tf.shape(output)[2]

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

        t2 = tf.squeeze(argmax)
        t2 = tf.stack((t2[0], t2[1]), axis=0)
        t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

        t = tf.concat([t2, t1], 3)
        indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

        x1 = tf.squeeze(x)
        x1 = tf.reshape(x1, [-1, channels])
        x1 = tf.transpose(x1, perm=[1, 0])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
        return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)

    def __call__(self,x):
        return self.unpool(x,self.raveled_argmax,self.name)
