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
        else:
            raise Exception(
                "Don't know about this '%s' layer type." % layer_type)
        return layer_name, layer



    print 'Loading VGG Net...'
    #network_layers = OrderedDict(struct_to_layer(network_params['layers'][0, i][
     #                            0, 0]) for i in xrange(network_params['layers'].shape[1]))
    network_layers = OrderedDict()
    for i in xrange(network_params['layers'].shape[1]):
        layer_name, layer = struct_to_layer(network_params['layers'][0, i][
                                 0, 0])
        
        network_layers[layer_name+'_layer'] = layer
        
        if up_to_layer == layer_name:
            break
                                                       
    # if up_to_layer is not None:
    #     if isinstance(up_to_layer, (list, tuple)):
    #         up_to_layer = network_layers.keys()[max(
    #             network_layers.keys().index(layer_name) for layer_name in up_to_layer)]
    #     layer_names = [network_params['layers'][0, i][0, 0][0][0]
    #                    for i in xrange(network_params['layers'].shape[1])]
    #     network_layers = OrderedDict((k, network_layers[k]) for k in layer_names[
    #                                  :layer_names.index(up_to_layer) + 1])
    print 'Done.'
    return ConvNet(network_layers)
