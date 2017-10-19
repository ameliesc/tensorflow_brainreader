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
            padding = 0 if layer_name.startswith('fc') else 1 if layer_name.startswith(
                'conv') else bad_value(layer_name)
            layer = # ConvLayer(w, b, force_shared_parameters=force_shared_parameters,
                              border_mode=padding, filter_flip=True)
        elif layer_type in ('relu', 'softmax'):
            layer = # Nonlinearity(layer_type)
        elif layer_type == 'pool':
            layer  = # Pooler(region=tuple(struct[3][0].astype(int)), stride=tuple(struct[4][0].astype(int)), mode=pooling_mode)
            switches = # Switches(region=tuple(struct[3][0].astype(int)), stride=tuple(struct[4][0].astype(int)))
        else:
            raise Exception(
                "Don't know about this '%s' layer type." % layer_type)
        return layer_name, layer, switches

    ## change commented functions above into tensorflow
