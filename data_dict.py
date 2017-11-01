from scipy.io import loadmat
from collections import OrderedDict


def data_dict(file_path='imagenet-vgg-verydeep-19.mat'):
    """
    : filepath: path to the vgg19.mat file
    : returns: a dictionary containing [weights,bias] for eachs layer_name
    """
    layer_params = OrderedDict()
    network_params = loadmat(file_path)
    for i in range(network_params['layers'].shape[1]):
        struct = network_params['layers'][0, i][0, 0]

        layer_type = struct[1][0]
        layer_name = str(struct[0][0])
         # OrderedDict()
        if layer_type == 'conv':
            w = struct[2][0, 0]
            b = struct[2][0, 1][:, 0]
            layer_params[layer_name] = [w,b]

    return layer_params


def test_data_ditc():
    a = data_dict()
    assert a['conv1_1'][0].shape == (3, 3, 3, 64)
    assert a['conv1_1'][1].shape == (64,)
