from scipy.io import loadmat
import numpy as np
import tensorflow as tf

def get_vgg_net():
    """ """
    #to do, download network automatically if not on computer yet, - find entwork on system

    network_params = loadmat('imagenet-vgg-verydeep-19.mat')

    def struct_to_layer(struct):
        layer_type = struct[1][0]
        layer_name = str(struct[0][0])
        switches = None
        assert isinstance(layer_type, basestring)
        if layer_type == 'conv':
            w_orig = struct[2][0, 0]  # (n_rows, n_cols, n_in_maps, n_out_maps)
            # (n_out_maps, n_in_maps, n_rows, n_cols)  (Theano conventions)
            w = w_orig.T.swapaxes(2, 3)
            b = struct[2][0, 1][:, 0]
            padding = 'VALID' if layer_name.startswith('fc') else 'SAME' if layer_name.startswith(
                'conv') else bad_value(layer_name)
            strides = struct[4][0]
            layer = ConvLayer(w, b, strides = strides, padding = padding,
                              border_mode=padding, name =  layer_name)
        elif layer_type in ('relu', 'softmax'):
            layer = Nonlinearity(layer_type, name = layer_name)
        elif layer_type == 'pool':
            pooling_mode = str(struct[2][0])
            layer  = Pooler(region=struct[3][0].tolist(), stride=struct[4][0].tolost(), mode=pooling_mode, name = layer_name)
           # switches = # Switches(region=tuple(struct[3][0].astype(int)), stride=tuple(struct[4][0].astype(int)))
        else:
            raise Exception(
                "Don't know about this '%s' layer type." % layer_type)
        return layer_name, layer

    ## change commented functions above into tensorflow
