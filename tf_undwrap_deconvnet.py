from collections import OrderedDict
from deconvnets import  Nonlinearity, Unpooler, Deconv, DeconvNet
from general.should_be_builtins import bad_value
from tf_makedeconvnet import load_conv_and_deconv
import pickle

def get_deconv( ind ,network_params = None,from_layer = None, force_shared_parameters=True):

    try:
        inn = open('deconvnetwork.p', 'r')
        network_params = pickle.load(network_params, inn)
        inn.close()
    except:
        network_params = load_conv_and_deconv()


    def struct_to_layer(struct, ind = None):
        for i in xrange(1,network_params['layers'].shape[1]): # first layer is softmax which is skipped
            layer_type = struct[1][0]
            layer_name = str(struct[0][0])
            if layer_type == 'deco':
                w_orig = struct[2][0, 0] 
                w = w_orig.T.swapaxes(2, 3)
                b = struct[2][0, 1][:, 0]
                strides = struct[4][0].tolist()
                padding = 'VALID' if layer_name.startswith('fc') else 'SAME' if layer_name.startswith('conv') else bad_value(layer_name)
                layer = Deconv(w, b, strides = strides, padding = padding, name = layer_name)
            elif layer_type == 'relu':
                layer = Nonlinearity(layer_type, name = layer_name)
            elif layer_type == 'unpo':
                indexes = ind[layer_name+'_indexes']
                pooling_mode = str(struct[2][0])
                layer = Unpooler(region=struct[3][0].tolist(), stride=struct[4][0].tolist(), mode=pooling_mode, indexes = indexes, name = layer_name)
            else:
                raise Exception(
                "Don't know about this '%s' layer type." % layer_type)
        return layer
    print "Loading DeconvNet..."
    network_layers = OrderedDict()

    for i in range(1,network_params['layers'].shape[1]):
        layer_name = str(network_params['layers'][0,i][0,0][0][0])
        if from_layer is not None: #dont understand what this does
            if from_layer != layer_name:
                continue
            else: 
                from_layer = None
        layer = struct_to_layer(network_params['layers'][0,i][0,0],ind)
        network_layers[layer_name] = layer
    print 'Done.'
    return DeconvNet(network_layers)
