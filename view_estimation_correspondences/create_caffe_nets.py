import numpy as np
import os
import sys
from warnings import warn
import tempfile
import shutil
from collections import OrderedDict
import time
import tempfile
import subprocess
import pdb
import argparse

# Import global variables
view_estimation_correspondences_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(view_estimation_correspondences_path))
import global_variables as gv

# Import Caffe
sys.path.append(gv.g_pycaffe_path)
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from caffe import layers as L
from caffe import params as P

# Define paths
train_models_root_path = os.path.join(gv.g_render4cnn_root_folder, 'train')

# Define common fillers and parameters
DEFAULT_LRN_PARAM = dict(local_size=5, alpha=0.0001, beta=0.75)
DEFAULT_DECAY_PARAM = [dict(decay_mult=1), dict(decay_mult=0)]
NO_LEARN_PARAM = dict(lr_mult=0)
DEFAULT_WEIGHT_FILLER = dict(type='gaussian', std=0.01)
DEFAULT_BIAS_FILLER = dict(type='constant', value=0)
DEFAULT_DROPOUT_RATIO = 0.5
DEFAULT_ANGLE_NAMES = ['azimuth', 'elevation', 'tilt']
DEFAULT_LOSS_WEIGHTS = [1, 1, 1]
BATCH_SIZE = 192

# Parameters for angle softmax+loss
DEFAULT_SOFTMAX_VIEW_LOSS_PARAM_A = dict(bandwidth=15, sigma=5, pos_weight=1, neg_weight=0, period=360)
DEFAULT_SOFTMAX_VIEW_LOSS_PARAM_ET = dict(bandwidth=5, sigma=3, pos_weight=1, neg_weight=0, period=360)
DEFAULT_VIEW_LOSS_PARAMS = [DEFAULT_SOFTMAX_VIEW_LOSS_PARAM_A, DEFAULT_SOFTMAX_VIEW_LOSS_PARAM_ET, DEFAULT_SOFTMAX_VIEW_LOSS_PARAM_ET]
# Parameters for angle accuracy
DEFAULT_ACCURACY_VIEW_PARAM_A = dict(tol_angle=15, period=360)
DEFAULT_ACCURACY_VIEW_PARAM_ET = dict(tol_angle=5, period=360)
DEFAULT_ACCURACY_VIEW_PARAMS = [DEFAULT_ACCURACY_VIEW_PARAM_A, DEFAULT_ACCURACY_VIEW_PARAM_ET, DEFAULT_ACCURACY_VIEW_PARAM_ET]

# Default solver
DEFAULT_SOLVER_DICT = dict(
    train_net=None,
    test_net=None,
    test_iter=15,
    test_interval=2000,
    base_lr=0.0001,
    lr_policy='fixed',
    max_iter=150000,
    display=20,
    momentum=0.9,
    momentum2=0.999,
    weight_decay=0.0005,
    snapshot=2000,
    snapshot_prefix='snapshot',
    solver_mode='GPU',
    type='Adam'
)


'''
Merge the given dictionaries and return the result.
@args:
    dict_args (args of type dict): The dictionaries to merge together.
'''
def merge_dicts(*dict_args):
    res = dict()
    for dict_arg in dict_args:
        res.update(dict_arg)
    return res

'''
Generate solver text from a dictionary.
@args
    d (dict): The dictionary to generate solver text from.
'''
def dict_to_solver_text(d, allow_empty=False):
    ret = ''
    for key, value in d.iteritems():
        if value is None:
            if allow_empty:
                warn('Generating solver with empty parameter %s' % key)
            else:
                raise Exception('Solver dictionary has empty parameter %s' % key)
        # Figure out if the value needs quotes around it. Strings generally need quotes, except for some stupid cases
        if isinstance(value, basestring) and key not in ['solver_mode']:
            value = '"%s"' % value
        ret += '%s: %s\n' % (key, value)
    return ret

'''
Create an in-place ReLU layer.
@args
    name (str): Name of the ReLU layer
    bottom (str): Name of the blob to apply ReLU to
'''
def relu(name, bottom):
    return L.ReLU(name=name, bottom=bottom, top=bottom, in_place=True)

'''
Create an in-place tanh layer.
@args
    name (str): Name of the tanh layer
    bottom (str): Name of the blob to apply tanh to
'''
def tanh(name, bottom):
    return L.TanH(name=name, bottom=bottom, top=bottom, in_place=True)

'''
Create an in-place Dropout layer.
@args
    name (str): Name of the Dropout layer
    bottom (str): Name of the blob to apply Dropout to
    dropout_ratio (float): How often to drop out the activation
'''
def dropout(name, bottom, dropout_ratio=DEFAULT_DROPOUT_RATIO):
    return L.Dropout(name=name, bottom=bottom, top=bottom, in_place=True, dropout_param=dict(
        dropout_ratio=dropout_ratio
    ))

'''
Create an InnerProduct (FC) layer with given filler and decay parameters.
@args
    name (str): Name of the InnerProduct layer
    bottom (str): Name of the input blob for the InnerProduct layer
    num_output (int): Number of outputs for the InnerProduct layer
    weight_filler (dict): The parameters for the weight filler
    bias_filler (dict): The parameters of the bias filler
'''
def innerproduct(name, bottom, num_output, param, weight_filler=DEFAULT_WEIGHT_FILLER, bias_filler=DEFAULT_BIAS_FILLER):
    return L.InnerProduct(name=name, bottom=bottom, param=param, inner_product_param=dict(
        num_output=num_output, weight_filler=weight_filler, bias_filler=bias_filler
    ))

'''
Create a LRN layer with given parameters.
@args
    name (str): Name of the LRN layer
    bottom (str): Name of the input blob for the LRN layer

'''
def lrn(name, bottom, lrn_param=DEFAULT_LRN_PARAM):
    return L.LRN(name=name, bottom=bottom, lrn_param=lrn_param)

'''
Augment the given network specification with a split layer with given input and output
@args
    net_spec (caffe.NetSpec): The network specification to augment
    bottom (str): Name of the input blob for the layer
    top (list of str): List of output blob names
'''
def add_split_layer(net_spec, bottom, top):
    net_spec[top[0]] = L.Split(bottom=bottom, top=top[1:])

'''
Augment the given network specification with a slice layer with given input and output
@args
    net_spec (caffe.NetSpec): The network specification to augment
    bottom (str): Name of the input blob for the layer
    top (list of str): List of output blob names
    slice_param (dict): The slice parameters
'''
def add_slice_layer(net_spec, bottom, top, slice_param):
    net_spec[top[0]] = L.Slice(bottom=bottom, top=top[1:], slice_param=slice_param)

'''
Augment the given network specification with a conv layer with optional activation wrappers. Layers are automatically generated based on the name of the base conv layer.
The wrappers are generated in order and named as follows: conv## -> relu## -> pool## -> norm##
@args
    net_spec (caffe.NetSpec): The network specification to augment
    name (str): Name of the base conv layer
    bottom (str): Name of the input blob for the base conv layer
    use_relu (bool): Whether to apply ReLU activation to the base conv layer
    pooling_param (dict): Parameters for the pooling layer, if desired
    lrn_param (dict): Parameters for the LRN layer, if desired
'''
def add_wrapped_conv_layer(net_spec, name, bottom, conv_param, param=None, use_relu=True, pooling_param=None, lrn_param=None):
    assert(name[:4] == 'conv')
    if param:
        net_spec[name] = L.Convolution(name=name, bottom=bottom, param=param, convolution_param=conv_param)
    else:
        net_spec[name] = L.Convolution(name=name, bottom=bottom, convolution_param=conv_param)
    out_name = name
    if use_relu:
        relu_name = name.replace('conv', 'relu')
        net_spec[relu_name] = relu(relu_name, name)
        out_name = name
    if pooling_param:
        pool_name = name.replace('conv', 'pool')
        net_spec[pool_name] = L.Pooling(name=pool_name, bottom=name, pooling_param=pooling_param)
        out_name = pool_name
    if lrn_param:
        lrn_name = name.replace('conv', 'norm')
        net_spec[lrn_name] = lrn(lrn_name, out_name, lrn_param=lrn_param)
        out_name = lrn_name
    return out_name

'''
Augment the given network specification with an InnerProduct (FC) layer with optional activation wrappers. Layers are automatically generated based on the name of the base InnerProduct layer.
The wrappers are generated in order and named as follows: fc## -> relu## -> drop##
@args
    net_spec (caffe.NetSpec): The network specification to augment
    name (str): Name of the base conv layer
    bottom (str): Name of the input blob for the base FC layer
    num_output (int): The number of outputs for the base FC layer
    use_relu (bool): Whether to apply ReLU activation to the base FC layer
    dropout_ratio (float): The dropout ratio, if desired
'''
def add_wrapped_fc_layer(net_spec, name, bottom, num_output, param=DEFAULT_DECAY_PARAM, use_relu=True, dropout_ratio=-1):
    assert(name[:2] == 'fc')
    net_spec[name] = innerproduct(name, bottom, num_output, param)
    if use_relu:
        relu_name = name.replace('fc', 'relu')
        net_spec[relu_name] = relu(relu_name, name)
    if dropout_ratio >= 0:
        dropout_name = name.replace('fc', 'drop')
        net_spec[dropout_name] = dropout(dropout_name, name)
    return name

'''
Augment the given network specification with prediction layers.
@args
    net_spec (caffe.NetSpec): The network specification augment
    name_prefix (str): The string to append to each output
    bottom (str): Name of the input blob for the prediction layers
    num_output (int): The number of outputs for each prediction layer
    angle_names (str): The names of the angles to predict
'''
def add_prediction_layers(net_spec, name_prefix, bottom, num_output=4320, param=DEFAULT_DECAY_PARAM, angle_names=DEFAULT_ANGLE_NAMES):
    for angle_name in angle_names:
        pred_name = name_prefix + angle_name
        net_spec[pred_name] = innerproduct(pred_name, bottom, num_output, param)

'''
Augment the given network specification with loss and accuracy layers.
@args
    net_spec (caffe.NetSpec): The network specification augment
    name_prefix (str): The string to append to each output
    bottom (str): Name of the input blob for the prediction layers
    num_output (int): The number of outputs for each prediction layer
    angle_names (str): The names of the angles to predict
'''
def add_loss_acc_layers(net_spec, bottom_prefixes, angle_names=DEFAULT_ANGLE_NAMES, loss_weights=DEFAULT_LOSS_WEIGHTS, loss_param_arr=DEFAULT_VIEW_LOSS_PARAMS, acc_param_arr=DEFAULT_ACCURACY_VIEW_PARAMS):
    assert(len(angle_names) == len(loss_weights) == len(loss_param_arr) == len(acc_param_arr))
    for i, angle_name in enumerate(angle_names):
        # Add loss layer for current angle
        loss_name = 'loss_' + angle_name
        bottom = [x + angle_name for x in bottom_prefixes]
        net_spec[loss_name] = L.SoftmaxWithViewLoss(name=loss_name, bottom=bottom, loss_weight=loss_weights[i], softmax_with_view_loss_param=loss_param_arr[i])
        # Add accuracy layer for current angle
        acc_name = 'accuracy_' + angle_name
        net_spec[acc_name] = L.AccuracyView(name=acc_name, bottom=bottom, accuracy_view_param=acc_param_arr[i])

'''
Get the model prototext string from a Caffe NetSpec.
@args
    net_spec (caffe.NetSpec): The network specification argument
'''
def get_prototxt_str(net_spec):
    return str(net_spec.to_proto())

'''
Test the validity of the given network parameters.
@args
    net_param (caffe.NetParameter): The network parameters to test
'''
def verify_netspec(net_param, net_weights_path=None):
    temp_file = tempfile.NamedTemporaryFile()
    # Write prototxt to file, and flush to make sure whole prototxt is written
    temp_file.write(str(net_param))
    temp_file.flush()
    # Try to make the net. Caffe will crash everything if it fails. This cannot be caught.
    if not net_weights_path:
        caffe.Net(temp_file.name, caffe.TRAIN)
    else:
        caffe.Net(temp_file.name, net_weights_path, caffe.TRAIN)

'''
Delete a layer from a NetParameter by its type.
@args
    net_param (caffe.NetParameter): The network parameter object to modify
    type_name (str): The type of the layers to delete
'''
def delete_layer_by_type(net_param, type_name):
    # Keep track of the names of deleted layers
    deleted_layer_names = []
    # Go through layers backwards to prevent index inconsistency
    layers = net_param.layer._values
    for i in reversed(range(len(layers))):
        if layers[i].type == type_name:
            layer = layers.pop(i)
            deleted_layer_names.insert(0, layer.name)
    return deleted_layer_names

'''
Delete a layer from a NetParameter by its name.
@args
    net_param (caffe.NetParameter): The network parameter object to modify
    layer_name (str): The name of the layer to delete
'''
def delete_layer_by_name(net_param, layer_name):
    layers = net_param.layer._values
    for i in range(len(layers)):
        if layers[i].name == layer_name:
            layers.pop(i)
            return

'''
Create a deployment prototxt string from a NetParameter. This deletes "Data" layers and replaces them with "input" and "input_shape" fields, and deletes view softmax and accuracy layers.
@args
    net_param (caffe.NetParameter): The network parameters to generate a deploy prototxt from
    data_shapes (dict<str, tuple>): A map from the layer name to the input shape
'''
def netspec_to_deploy_prototxt_str(net_param, data_shapes):
    ret = ''
    # Get data layer names
    layers = net_param.layer._values
    data_layer_names = [layer.name for layer in layers if layer.type == 'Data']
    # Add input layer specification if it is in both the net parameters and the data shapes
    for name, shape in data_shapes.iteritems():
        if name in data_layer_names:
            ret += 'input: "%s"\ninput_shape {\n' % name
            for dim_size in shape:
                ret += '  dim: %d\n' % dim_size
            ret += '}\n'
        else:
            warn('netspec_to_deploy_prototxt_str: Skipping unavailable data layer ' + name)
    # Create copy of net_param to modify
    net_param_copy = caffe_pb2.NetParameter()
    text_format.Merge(str(net_param), net_param_copy)
    # Delete data, softmax, and accuracy layers
    for layer_type in ['Data', 'Silence', 'SoftmaxWithViewLoss', 'AccuracyView']:
        delete_layer_by_type(net_param_copy, layer_type)
    # Delete label slice layer
    delete_layer_by_name(net_param_copy, 'labe-slice')
    ret += str(net_param_copy)
    return ret

'''
R4CNN
'''
def create_model_r4cnn(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    train_data_lmdb_path = lmdb_paths[0]
    train_label_lmdb_path = lmdb_paths[1]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=train_data_lmdb_path, transform_param=data_transform_param)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=train_label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # Image (Render for CNN) features
    conv1_param = dict(num_output=96, kernel_size=11, stride=4)
    pool1_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2_param = dict(num_output=256, pad=2, kernel_size=5, group=2)
    pool2_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv3_param = dict(num_output=384, pad=1, kernel_size=3)
    conv4_param = dict(num_output=384, pad=1, kernel_size=3, group=2)
    conv5_param = dict(num_output=256, pad=1, kernel_size=3, group=2)
    pool5_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_param, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_param, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_param)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_param, param=DEFAULT_DECAY_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_param, param=DEFAULT_DECAY_PARAM, pooling_param=pool5_param)
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6', conv5_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7', fc6_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'fc-', fc7_out_name)
    add_loss_acc_layers(n, ['fc-', 'label_'])

    return n.to_proto()


def create_model_chcnn_chessboard(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    train_data_lmdb_path = lmdb_paths[0]
    keypoint_map_lmdb_path = lmdb_paths[1]
    keypoint_class_lmdb_path = lmdb_paths[2]
    train_label_lmdb_path = lmdb_paths[3]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=train_data_lmdb_path, transform_param=data_transform_param)
    n['keypoint_map'] = L.Data(name='keypoint_map', batch_size=batch_size, backend=P.Data.LMDB, source=keypoint_map_lmdb_path)
    n['keypoint_class'] = L.Data(name='keypoint_class', batch_size=batch_size, backend=P.Data.LMDB, source=keypoint_class_lmdb_path)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=train_label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # Image (Render for CNN) features
    conv1_param = dict(num_output=96, kernel_size=11, stride=4)
    pool1_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2_param = dict(num_output=256, pad=2, kernel_size=5, group=2)
    pool2_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv3_param = dict(num_output=384, pad=1, kernel_size=3)
    conv4_param = dict(num_output=384, pad=1, kernel_size=3, group=2)
    conv5_param = dict(num_output=256, pad=1, kernel_size=3, group=2)
    pool5_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_param, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_param, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_param)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_param, param=DEFAULT_DECAY_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_param, param=DEFAULT_DECAY_PARAM, pooling_param=pool5_param)
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6', conv5_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7', fc6_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Keypoint map features
    scale_keypoint_map_param = dict(filler=dict(value=1/227.0))
    pool_keypoint_map_param = dict(pool=P.Pooling.MAX, kernel_size=5, stride=5, pad=3)
    reshape_keypoint_map_param = dict(shape=dict(dim=[0, 2116, 1, 1]))
    n['scale-keypoint-map'] = L.Scale(name='scale-keypoint-map', bottom='keypoint_map', param=NO_LEARN_PARAM, scale_param=scale_keypoint_map_param)
    n['pool-keypoint-map'] = L.Pooling(name='pool-keypoint-map', bottom='scale-keypoint-map', pooling_param=pool_keypoint_map_param)
    fc_keypoint_map_name = add_wrapped_fc_layer(n, 'fc-keypoint-map', 'pool-keypoint-map', 2116, use_relu=False)

    # Keypoint class features
    fc_keypoint_class_name = add_wrapped_fc_layer(n, 'fc-keypoint-class', 'keypoint_class', 34, use_relu=False)

    # Attention map
    # keypoint_concat_param = dict(axis=1)
    attn_param = dict(shape=dict(dim=[0, 1, 13, 13]))
    n['keypoint-concat'] = L.Concat(name='keypoint-concat', bottom=[fc_keypoint_map_name, fc_keypoint_class_name])
    fc_keypoint_concat_name = add_wrapped_fc_layer(n, 'fc-keypoint-concat', 'keypoint-concat', 169, use_relu=False)
    n['fc-keypoint-concat-softmax'] = L.Softmax(name='fc-keypoint-concat-softmax', bottom=fc_keypoint_concat_name)
    n['attn'] = L.Reshape(name='attn', bottom='fc-keypoint-concat-softmax', reshape_param=attn_param)

    # Get conv4 attention vector. First, slice channels
    add_slice_layer(n, 'conv4', ['conv4_slice_' + str(i) for i in range(384)], dict(
        axis=1, slice_point=range(1, 384)
    ))
    # Then weigh each channel by the map
    for i in range(384):
        slice_bottom_name = 'conv4_slice_' + str(i)
        top_name = 'conv4_weighted_' + str(i)
        n[top_name] = L.Eltwise(name=top_name, bottom=[slice_bottom_name, 'attn'], eltwise_param=dict(operation=P.Eltwise.PROD))
    # Combine weighted maps
    n['conv4_weighted'] = L.Concat(name='conv4_weighted', bottom=['conv4_weighted_' + str(i) for i in range(384)])
    # Sum across width
    add_slice_layer(n, 'conv4_weighted', ['conv4_weighted_slice_' + str(i) for i in range(13)], dict(
        axis=3, slice_point=range(1, 13)
    ))
    n['conv4_weighted_a'] = L.Eltwise(name='conv4_weighted_a', bottom=['conv4_weighted_slice_' + str(i) for i in range(13)], eltwise_param=dict(operation=P.Eltwise.SUM))
    # Sum across height
    add_slice_layer(n, 'conv4_weighted_a', ['conv4_weighted_a_slice_' + str(i) for i in range(13)], dict(
        axis=2, slice_point=range(1, 13)
    ))
    n['conv4_weighted_b'] = L.Eltwise(name='conv4_weighted_a', bottom=['conv4_weighted_a_slice_' + str(i) for i in range(13)], eltwise_param=dict(operation=P.Eltwise.SUM))
    # Finally, flatten to get attention vector
    n['conv4-a'] = L.Flatten(name='conv4-a', bottom='conv4_weighted_b')

    # Fusion
    n['fc7-concat'] = L.Concat(name='fc7-concat', bottom=[fc7_out_name, 'conv4-a'])
    fc8_out_name = add_wrapped_fc_layer(n, 'fc8', 'fc7-concat', 4096, param=DEFAULT_DECAY_PARAM, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc8_out_name, param=DEFAULT_DECAY_PARAM)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()

def create_model_chcnn_manhattan(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    train_data_lmdb_path = lmdb_paths[0]
    keypoint_map_lmdb_path = lmdb_paths[1]
    keypoint_class_lmdb_path = lmdb_paths[2]
    train_label_lmdb_path = lmdb_paths[3]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=train_data_lmdb_path, transform_param=data_transform_param)
    n['keypoint_map'] = L.Data(name='keypoint_map', batch_size=batch_size, backend=P.Data.LMDB, source=keypoint_map_lmdb_path)
    n['keypoint_class'] = L.Data(name='keypoint_class', batch_size=batch_size, backend=P.Data.LMDB, source=keypoint_class_lmdb_path)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=train_label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # Image (Render for CNN) features
    conv1_param = dict(num_output=96, kernel_size=11, stride=4)
    pool1_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2_param = dict(num_output=256, pad=2, kernel_size=5, group=2)
    pool2_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv3_param = dict(num_output=384, pad=1, kernel_size=3)
    conv4_param = dict(num_output=384, pad=1, kernel_size=3, group=2)
    conv5_param = dict(num_output=256, pad=1, kernel_size=3, group=2)
    pool5_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_param, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_param, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_param)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_param, param=DEFAULT_DECAY_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_param, param=DEFAULT_DECAY_PARAM, pooling_param=pool5_param)
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6', conv5_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7', fc6_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Keypoint map features
    scale_keypoint_map_param = dict(filler=dict(value=1/(227.0 * 2)))
    pool_keypoint_map_param = dict(pool=P.Pooling.MAX, kernel_size=5, stride=5, pad=3)
    n['scale-keypoint-map'] = L.Scale(name='scale-keypoint-map', bottom='keypoint_map', param=NO_LEARN_PARAM, scale_param=scale_keypoint_map_param)
    n['pool-keypoint-map'] = L.Pooling(name='pool-keypoint-map', bottom='scale-keypoint-map', pooling_param=pool_keypoint_map_param)
    fc_keypoint_map_name = add_wrapped_fc_layer(n, 'fc-keypoint-map', 'pool-keypoint-map', 2116, use_relu=False)

    # Keypoint class features
    fc_keypoint_class_name = add_wrapped_fc_layer(n, 'fc-keypoint-class', 'keypoint_class', 34, use_relu=False)

    # Attention map
    # keypoint_concat_param = dict(axis=1)
    attn_param = dict(shape=dict(dim=[0, 1, 13, 13]))
    n['keypoint-concat'] = L.Concat(name='keypoint-concat', bottom=[fc_keypoint_map_name, fc_keypoint_class_name])
    fc_keypoint_concat_name = add_wrapped_fc_layer(n, 'fc-keypoint-concat', 'keypoint-concat', 169, use_relu=False)
    n['fc-keypoint-concat-softmax'] = L.Softmax(name='fc-keypoint-concat-softmax', bottom=fc_keypoint_concat_name)
    n['attn'] = L.Reshape(name='attn', bottom='fc-keypoint-concat-softmax', reshape_param=attn_param)

    # Get conv4 attention vector. First, slice channels
    add_slice_layer(n, 'conv4', ['conv4_slice_' + str(i) for i in range(384)], dict(
        axis=1, slice_point=range(1, 384)
    ))
    # Then weigh each channel by the map
    for i in range(384):
        slice_bottom_name = 'conv4_slice_' + str(i)
        top_name = 'conv4_weighted_' + str(i)
        n[top_name] = L.Eltwise(name=top_name, bottom=[slice_bottom_name, 'attn'], eltwise_param=dict(operation=P.Eltwise.PROD))
    # Combine weighted maps
    n['conv4_weighted'] = L.Concat(name='conv4_weighted', bottom=['conv4_weighted_' + str(i) for i in range(384)])
    # Sum across width
    add_slice_layer(n, 'conv4_weighted', ['conv4_weighted_slice_' + str(i) for i in range(13)], dict(
        axis=3, slice_point=range(1, 13)
    ))
    n['conv4_weighted_a'] = L.Eltwise(name='conv4_weighted_a', bottom=['conv4_weighted_slice_' + str(i) for i in range(13)], eltwise_param=dict(operation=P.Eltwise.SUM))
    # Sum across height
    add_slice_layer(n, 'conv4_weighted_a', ['conv4_weighted_a_slice_' + str(i) for i in range(13)], dict(
        axis=2, slice_point=range(1, 13)
    ))
    n['conv4_weighted_b'] = L.Eltwise(name='conv4_weighted_a', bottom=['conv4_weighted_a_slice_' + str(i) for i in range(13)], eltwise_param=dict(operation=P.Eltwise.SUM))
    # Finally, flatten to get attention vector
    n['conv4-a'] = L.Flatten(name='conv4-a', bottom='conv4_weighted_b')

    # Fusion
    n['fc7-concat'] = L.Concat(name='fc7-concat', bottom=[fc7_out_name, 'conv4-a'])
    fc8_out_name = add_wrapped_fc_layer(n, 'fc8', 'fc7-concat', 4096, param=DEFAULT_DECAY_PARAM, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc8_out_name, param=DEFAULT_DECAY_PARAM)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()

def create_model_chcnn_euclidean(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    train_data_lmdb_path = lmdb_paths[0]
    keypoint_map_lmdb_path = lmdb_paths[1]
    keypoint_class_lmdb_path = lmdb_paths[2]
    train_label_lmdb_path = lmdb_paths[3]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=train_data_lmdb_path, transform_param=data_transform_param)
    n['keypoint_map'] = L.Data(name='keypoint_map', batch_size=batch_size, backend=P.Data.LMDB, source=keypoint_map_lmdb_path)
    n['keypoint_class'] = L.Data(name='keypoint_class', batch_size=batch_size, backend=P.Data.LMDB, source=keypoint_class_lmdb_path)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=train_label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # Image (Render for CNN) features
    conv1_param = dict(num_output=96, kernel_size=11, stride=4)
    pool1_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2_param = dict(num_output=256, pad=2, kernel_size=5, group=2)
    pool2_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv3_param = dict(num_output=384, pad=1, kernel_size=3)
    conv4_param = dict(num_output=384, pad=1, kernel_size=3, group=2)
    conv5_param = dict(num_output=256, pad=1, kernel_size=3, group=2)
    pool5_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_param, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_param, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_param)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_param, param=DEFAULT_DECAY_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_param, param=DEFAULT_DECAY_PARAM, pooling_param=pool5_param)
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6', conv5_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7', fc6_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Keypoint map features
    scale_keypoint_map_param = dict(filler=dict(value=1/(227.0 * np.sqrt(2))))
    pool_keypoint_map_param = dict(pool=P.Pooling.MAX, kernel_size=5, stride=5, pad=3)
    reshape_keypoint_map_param = dict(shape=dict(dim=[0, 2116, 1, 1]))
    n['scale-keypoint-map'] = L.Scale(name='scale-keypoint-map', bottom='keypoint_map', param=NO_LEARN_PARAM, scale_param=scale_keypoint_map_param)
    n['pool-keypoint-map'] = L.Pooling(name='pool-keypoint-map', bottom='scale-keypoint-map', pooling_param=pool_keypoint_map_param)
    fc_keypoint_map_name = add_wrapped_fc_layer(n, 'fc-keypoint-map', 'pool-keypoint-map', 2116, use_relu=False)

    # Keypoint class features
    fc_keypoint_class_name = add_wrapped_fc_layer(n, 'fc-keypoint-class', 'keypoint_class', 34, use_relu=False)

    # Attention map
    # keypoint_concat_param = dict(axis=1)
    attn_param = dict(shape=dict(dim=[0, 1, 13, 13]))
    n['keypoint-concat'] = L.Concat(name='keypoint-concat', bottom=[fc_keypoint_map_name, fc_keypoint_class_name])
    fc_keypoint_concat_name = add_wrapped_fc_layer(n, 'fc-keypoint-concat', 'keypoint-concat', 169, use_relu=False)
    n['fc-keypoint-concat-softmax'] = L.Softmax(name='fc-keypoint-concat-softmax', bottom=fc_keypoint_concat_name)
    n['attn'] = L.Reshape(name='attn', bottom='fc-keypoint-concat-softmax', reshape_param=attn_param)

    # Get conv4 attention vector. First, slice channels
    add_slice_layer(n, 'conv4', ['conv4_slice_' + str(i) for i in range(384)], dict(
        axis=1, slice_point=range(1, 384)
    ))
    # Then weigh each channel by the map
    for i in range(384):
        slice_bottom_name = 'conv4_slice_' + str(i)
        top_name = 'conv4_weighted_' + str(i)
        n[top_name] = L.Eltwise(name=top_name, bottom=[slice_bottom_name, 'attn'], eltwise_param=dict(operation=P.Eltwise.PROD))
    # Combine weighted maps
    n['conv4_weighted'] = L.Concat(name='conv4_weighted', bottom=['conv4_weighted_' + str(i) for i in range(384)])
    # Sum across width
    add_slice_layer(n, 'conv4_weighted', ['conv4_weighted_slice_' + str(i) for i in range(13)], dict(
        axis=3, slice_point=range(1, 13)
    ))
    n['conv4_weighted_a'] = L.Eltwise(name='conv4_weighted_a', bottom=['conv4_weighted_slice_' + str(i) for i in range(13)], eltwise_param=dict(operation=P.Eltwise.SUM))
    # Sum across height
    add_slice_layer(n, 'conv4_weighted_a', ['conv4_weighted_a_slice_' + str(i) for i in range(13)], dict(
        axis=2, slice_point=range(1, 13)
    ))
    n['conv4_weighted_b'] = L.Eltwise(name='conv4_weighted_a', bottom=['conv4_weighted_a_slice_' + str(i) for i in range(13)], eltwise_param=dict(operation=P.Eltwise.SUM))
    # Finally, flatten to get attention vector
    n['conv4-a'] = L.Flatten(name='conv4-a', bottom='conv4_weighted_b')

    # Fusion
    n['fc7-concat'] = L.Concat(name='fc7-concat', bottom=[fc7_out_name, 'conv4-a'])
    fc8_out_name = add_wrapped_fc_layer(n, 'fc8', 'fc7-concat', 4096, param=DEFAULT_DECAY_PARAM, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc8_out_name, param=DEFAULT_DECAY_PARAM)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()


'''
Gaussian keypoint map
'''
def create_model_chcnn_gaussian(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    train_data_lmdb_path = lmdb_paths[0]
    keypoint_map_lmdb_path = lmdb_paths[1]
    keypoint_class_lmdb_path = lmdb_paths[2]
    train_label_lmdb_path = lmdb_paths[3]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=train_data_lmdb_path, transform_param=data_transform_param)
    n['keypoint_map'] = L.Data(name='keypoint_map', batch_size=batch_size, backend=P.Data.LMDB, source=keypoint_map_lmdb_path)
    n['keypoint_class'] = L.Data(name='keypoint_class', batch_size=batch_size, backend=P.Data.LMDB, source=keypoint_class_lmdb_path)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=train_label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # Image (Render for CNN) features
    conv1_param = dict(num_output=96, kernel_size=11, stride=4)
    pool1_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2_param = dict(num_output=256, pad=2, kernel_size=5, group=2)
    pool2_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv3_param = dict(num_output=384, pad=1, kernel_size=3)
    conv4_param = dict(num_output=384, pad=1, kernel_size=3, group=2)
    conv5_param = dict(num_output=256, pad=1, kernel_size=3, group=2)
    pool5_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_param, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_param, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_param)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_param, param=DEFAULT_DECAY_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_param, param=DEFAULT_DECAY_PARAM, pooling_param=pool5_param)
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6', conv5_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7', fc6_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Keypoint map features
    pool_keypoint_map_param = dict(pool=P.Pooling.MAX, kernel_size=5, stride=5, pad=3)
    n['pool-keypoint-map'] = L.Pooling(name='pool-keypoint-map', bottom='keypoint_map', pooling_param=pool_keypoint_map_param)
    fc_keypoint_map_name = add_wrapped_fc_layer(n, 'fc-keypoint-map', 'pool-keypoint-map', 2116, use_relu=False)

    # Keypoint class features
    fc_keypoint_class_name = add_wrapped_fc_layer(n, 'fc-keypoint-class', 'keypoint_class', 34, use_relu=False)

    # Attention map
    # keypoint_concat_param = dict(axis=1)
    attn_param = dict(shape=dict(dim=[0, 1, 13, 13]))
    n['keypoint-concat'] = L.Concat(name='keypoint-concat', bottom=[fc_keypoint_map_name, fc_keypoint_class_name])
    fc_keypoint_concat_name = add_wrapped_fc_layer(n, 'fc-keypoint-concat', 'keypoint-concat', 169, use_relu=False)
    n['fc-keypoint-concat-softmax'] = L.Softmax(name='fc-keypoint-concat-softmax', bottom=fc_keypoint_concat_name)
    n['attn'] = L.Reshape(name='attn', bottom='fc-keypoint-concat-softmax', reshape_param=attn_param)

    # Get conv4 attention vector. First, slice channels
    add_slice_layer(n, 'conv4', ['conv4_slice_' + str(i) for i in range(384)], dict(
        axis=1, slice_point=range(1, 384)
    ))
    # Then weigh each channel by the map
    for i in range(384):
        slice_bottom_name = 'conv4_slice_' + str(i)
        top_name = 'conv4_weighted_' + str(i)
        n[top_name] = L.Eltwise(name=top_name, bottom=[slice_bottom_name, 'attn'], eltwise_param=dict(operation=P.Eltwise.PROD))
    # Combine weighted maps
    n['conv4_weighted'] = L.Concat(name='conv4_weighted', bottom=['conv4_weighted_' + str(i) for i in range(384)])
    # Sum across width
    add_slice_layer(n, 'conv4_weighted', ['conv4_weighted_slice_' + str(i) for i in range(13)], dict(
        axis=3, slice_point=range(1, 13)
    ))
    n['conv4_weighted_a'] = L.Eltwise(name='conv4_weighted_a', bottom=['conv4_weighted_slice_' + str(i) for i in range(13)], eltwise_param=dict(operation=P.Eltwise.SUM))
    # Sum across height
    add_slice_layer(n, 'conv4_weighted_a', ['conv4_weighted_a_slice_' + str(i) for i in range(13)], dict(
        axis=2, slice_point=range(1, 13)
    ))
    n['conv4_weighted_b'] = L.Eltwise(name='conv4_weighted_a', bottom=['conv4_weighted_a_slice_' + str(i) for i in range(13)], eltwise_param=dict(operation=P.Eltwise.SUM))
    # Finally, flatten to get attention vector
    n['conv4-a'] = L.Flatten(name='conv4-a', bottom='conv4_weighted_b')

    # Fusion
    n['fc7-concat'] = L.Concat(name='fc7-concat', bottom=[fc7_out_name, 'conv4-a'])
    fc8_out_name = add_wrapped_fc_layer(n, 'fc8', 'fc7-concat', 4096, param=DEFAULT_DECAY_PARAM, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc8_out_name, param=DEFAULT_DECAY_PARAM)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()


'''
Fixed uniform weight map
'''
def create_model_uniform_weights(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    train_data_lmdb_path = lmdb_paths[0]
    keypoint_map_lmdb_path = lmdb_paths[1]
    keypoint_class_lmdb_path = lmdb_paths[2]
    train_label_lmdb_path = lmdb_paths[3]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=train_data_lmdb_path, transform_param=data_transform_param)
    n['keypoint_map'] = L.Data(name='keypoint_map', batch_size=batch_size, backend=P.Data.LMDB, source=keypoint_map_lmdb_path)
    n['keypoint_class'] = L.Data(name='keypoint_class', batch_size=batch_size, backend=P.Data.LMDB, source=keypoint_class_lmdb_path)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=train_label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # Image (Render for CNN) features
    conv1_param = dict(num_output=96, kernel_size=11, stride=4)
    pool1_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2_param = dict(num_output=256, pad=2, kernel_size=5, group=2)
    pool2_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv3_param = dict(num_output=384, pad=1, kernel_size=3)
    conv4_param = dict(num_output=384, pad=1, kernel_size=3, group=2)
    conv5_param = dict(num_output=256, pad=1, kernel_size=3, group=2)
    pool5_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_param, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_param, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_param)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_param, param=DEFAULT_DECAY_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_param, param=DEFAULT_DECAY_PARAM, pooling_param=pool5_param)
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6', conv5_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7', fc6_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Keypoint map features
    scale_keypoint_map_param = dict(filler=dict(value=1/227.0))
    pool_keypoint_map_param = dict(pool=P.Pooling.MAX, kernel_size=5, stride=5, pad=3)
    reshape_keypoint_map_param = dict(shape=dict(dim=[0, 2116, 1, 1]))
    n['scale-keypoint-map'] = L.Scale(name='scale-keypoint-map', bottom='keypoint_map', param=NO_LEARN_PARAM, scale_param=scale_keypoint_map_param)
    n['pool-keypoint-map'] = L.Pooling(name='pool-keypoint-map', bottom='scale-keypoint-map', pooling_param=pool_keypoint_map_param)
    n['reshape-keypoint-map'] = L.Reshape(name='reshape-keypoint-map', bottom='pool-keypoint-map', reshape_param=reshape_keypoint_map_param)

    # Attention map
    keypoint_concat_param = dict(axis=1)
    attn_param = dict(shape=dict(dim=[0, 1, 13, 13]))
    constant_keypoint_concat_param = dict(filler=dict(value=0))
    n['keypoint-concat'] = L.Concat(name='keypoint-concat', bottom=['reshape-keypoint-map', 'keypoint_class'], concat_param=keypoint_concat_param)
    fc_keypoint_concat_name = add_wrapped_fc_layer(n, 'fc-keypoint-concat', 'keypoint-concat', 169, use_relu=False)
    n['constant-keypoint-concat'] = L.Scale(name='constant-keypoint-concat', bottom=fc_keypoint_concat_name, param=NO_LEARN_PARAM, scale_param=constant_keypoint_concat_param)
    n['fc-keypoint-concat-softmax'] = L.Softmax(name='fc-keypoint-concat-softmax', bottom='constant-keypoint-concat')
    n['attn'] = L.Reshape(name='attn', bottom='fc-keypoint-concat-softmax', reshape_param=attn_param)

    # Get conv4 attention vector. First, slice channels
    add_slice_layer(n, 'conv4', ['conv4_slice_' + str(i) for i in range(384)], dict(
        axis=1, slice_point=range(1, 384)
    ))
    # Then weigh each channel by the map
    for i in range(384):
        slice_bottom_name = 'conv4_slice_' + str(i)
        top_name = 'conv4_weighted_' + str(i)
        n[top_name] = L.Eltwise(name=top_name, bottom=[slice_bottom_name, 'attn'], eltwise_param=dict(operation=P.Eltwise.PROD))
    # Combine weighted maps
    n['conv4_weighted'] = L.Concat(name='conv4_weighted', bottom=['conv4_weighted_' + str(i) for i in range(384)])
    # Sum across width
    add_slice_layer(n, 'conv4_weighted', ['conv4_weighted_slice_' + str(i) for i in range(13)], dict(
        axis=3, slice_point=range(1, 13)
    ))
    n['conv4_weighted_a'] = L.Eltwise(name='conv4_weighted_a', bottom=['conv4_weighted_slice_' + str(i) for i in range(13)], eltwise_param=dict(operation=P.Eltwise.SUM))
    # Sum across height
    add_slice_layer(n, 'conv4_weighted_a', ['conv4_weighted_a_slice_' + str(i) for i in range(13)], dict(
        axis=2, slice_point=range(1, 13)
    ))
    n['conv4_weighted_b'] = L.Eltwise(name='conv4_weighted_a', bottom=['conv4_weighted_a_slice_' + str(i) for i in range(13)], eltwise_param=dict(operation=P.Eltwise.SUM))
    # Finally, flatten to get attention vector
    n['conv4-a'] = L.Flatten(name='conv4-a', bottom='conv4_weighted_b')

    # Fusion
    n['fc7-concat'] = L.Concat(name='fc7-concat', bottom=[fc7_out_name, 'conv4-a'])
    fc8_out_name = add_wrapped_fc_layer(n, 'fc8', 'fc7-concat', 4096, param=DEFAULT_DECAY_PARAM, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc8_out_name, param=DEFAULT_DECAY_PARAM)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()

'''
Fixed Gaussian weight map
'''
def create_model_gaussian_weights(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    train_data_lmdb_path = lmdb_paths[0]
    gaussian_attn_map_lmdb_path = lmdb_paths[1]
    train_label_lmdb_path = lmdb_paths[2]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=train_data_lmdb_path, transform_param=data_transform_param)
    n['attn'] = L.Data(name='attn', batch_size=batch_size, backend=P.Data.LMDB, source=gaussian_attn_map_lmdb_path)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=train_label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # Image (Render for CNN) features
    conv1_param = dict(num_output=96, kernel_size=11, stride=4)
    pool1_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2_param = dict(num_output=256, pad=2, kernel_size=5, group=2)
    pool2_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv3_param = dict(num_output=384, pad=1, kernel_size=3)
    conv4_param = dict(num_output=384, pad=1, kernel_size=3, group=2)
    conv5_param = dict(num_output=256, pad=1, kernel_size=3, group=2)
    pool5_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_param, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_param, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_param)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_param, param=DEFAULT_DECAY_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_param, param=DEFAULT_DECAY_PARAM, pooling_param=pool5_param)
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6', conv5_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7', fc6_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Get conv4 attention vector. First, slice channels
    add_slice_layer(n, 'conv4', ['conv4_slice_' + str(i) for i in range(384)], dict(
        axis=1, slice_point=range(1, 384)
    ))
    # Then weigh each channel by the map
    for i in range(384):
        slice_bottom_name = 'conv4_slice_' + str(i)
        top_name = 'conv4_weighted_' + str(i)
        n[top_name] = L.Eltwise(name=top_name, bottom=[slice_bottom_name, 'attn'], eltwise_param=dict(operation=P.Eltwise.PROD))
    # Combine weighted maps
    n['conv4_weighted'] = L.Concat(name='conv4_weighted', bottom=['conv4_weighted_' + str(i) for i in range(384)])
    # Sum across width
    add_slice_layer(n, 'conv4_weighted', ['conv4_weighted_slice_' + str(i) for i in range(13)], dict(
        axis=3, slice_point=range(1, 13)
    ))
    n['conv4_weighted_a'] = L.Eltwise(name='conv4_weighted_a', bottom=['conv4_weighted_slice_' + str(i) for i in range(13)], eltwise_param=dict(operation=P.Eltwise.SUM))
    # Sum across height
    add_slice_layer(n, 'conv4_weighted_a', ['conv4_weighted_a_slice_' + str(i) for i in range(13)], dict(
        axis=2, slice_point=range(1, 13)
    ))
    n['conv4_weighted_b'] = L.Eltwise(name='conv4_weighted_a', bottom=['conv4_weighted_a_slice_' + str(i) for i in range(13)], eltwise_param=dict(operation=P.Eltwise.SUM))
    # Finally, flatten to get attention vector
    n['conv4-a'] = L.Flatten(name='conv4-a', bottom='conv4_weighted_b')

    # Fusion
    n['fc7-concat'] = L.Concat(name='fc7-concat', bottom=[fc7_out_name, 'conv4-a'])
    fc8_out_name = add_wrapped_fc_layer(n, 'fc8', 'fc7-concat', 4096, param=DEFAULT_DECAY_PARAM, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc8_out_name, param=DEFAULT_DECAY_PARAM)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()


'''
Weight map learned from keypoint map only
'''
def create_model_chcnn_kpm_only(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    train_data_lmdb_path = lmdb_paths[0]
    keypoint_map_lmdb_path = lmdb_paths[1]
    train_label_lmdb_path = lmdb_paths[2]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=train_data_lmdb_path, transform_param=data_transform_param)
    n['keypoint_map'] = L.Data(name='keypoint_map', batch_size=batch_size, backend=P.Data.LMDB, source=keypoint_map_lmdb_path)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=train_label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # Image (Render for CNN) features
    conv1_param = dict(num_output=96, kernel_size=11, stride=4)
    pool1_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2_param = dict(num_output=256, pad=2, kernel_size=5, group=2)
    pool2_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv3_param = dict(num_output=384, pad=1, kernel_size=3)
    conv4_param = dict(num_output=384, pad=1, kernel_size=3, group=2)
    conv5_param = dict(num_output=256, pad=1, kernel_size=3, group=2)
    pool5_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_param, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_param, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_param)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_param, param=DEFAULT_DECAY_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_param, param=DEFAULT_DECAY_PARAM, pooling_param=pool5_param)
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6', conv5_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7', fc6_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Keypoint map features
    scale_keypoint_map_param = dict(filler=dict(value=1/227.0))
    pool_keypoint_map_param = dict(pool=P.Pooling.MAX, kernel_size=5, stride=5, pad=3)
    reshape_keypoint_map_param = dict(shape=dict(dim=[0, 2116, 1, 1]))
    n['scale-keypoint-map'] = L.Scale(name='scale-keypoint-map', bottom='keypoint_map', param=NO_LEARN_PARAM, scale_param=scale_keypoint_map_param)
    n['pool-keypoint-map'] = L.Pooling(name='pool-keypoint-map', bottom='scale-keypoint-map', pooling_param=pool_keypoint_map_param)
    fc_keypoint_map_name = add_wrapped_fc_layer(n, 'fc-keypoint-map', 'pool-keypoint-map', 2116, use_relu=False)

    # Attention map
    attn_param = dict(shape=dict(dim=[0, 1, 13, 13]))
    fc_keypoint_concat_name = add_wrapped_fc_layer(n, 'fc-keypoint-concat', fc_keypoint_map_name, 169, use_relu=False)
    n['fc-keypoint-concat-softmax'] = L.Softmax(name='fc-keypoint-concat-softmax', bottom=fc_keypoint_concat_name)
    n['attn'] = L.Reshape(name='attn', bottom='fc-keypoint-concat-softmax', reshape_param=attn_param)

    # Get conv4 attention vector. First, slice channels
    add_slice_layer(n, 'conv4', ['conv4_slice_' + str(i) for i in range(384)], dict(
        axis=1, slice_point=range(1, 384)
    ))
    # Then weigh each channel by the map
    for i in range(384):
        slice_bottom_name = 'conv4_slice_' + str(i)
        top_name = 'conv4_weighted_' + str(i)
        n[top_name] = L.Eltwise(name=top_name, bottom=[slice_bottom_name, 'attn'], eltwise_param=dict(operation=P.Eltwise.PROD))
    # Combine weighted maps
    n['conv4_weighted'] = L.Concat(name='conv4_weighted', bottom=['conv4_weighted_' + str(i) for i in range(384)])
    # Sum across width
    add_slice_layer(n, 'conv4_weighted', ['conv4_weighted_slice_' + str(i) for i in range(13)], dict(
        axis=3, slice_point=range(1, 13)
    ))
    n['conv4_weighted_a'] = L.Eltwise(name='conv4_weighted_a', bottom=['conv4_weighted_slice_' + str(i) for i in range(13)], eltwise_param=dict(operation=P.Eltwise.SUM))
    # Sum across height
    add_slice_layer(n, 'conv4_weighted_a', ['conv4_weighted_a_slice_' + str(i) for i in range(13)], dict(
        axis=2, slice_point=range(1, 13)
    ))
    n['conv4_weighted_b'] = L.Eltwise(name='conv4_weighted_a', bottom=['conv4_weighted_a_slice_' + str(i) for i in range(13)], eltwise_param=dict(operation=P.Eltwise.SUM))
    # Finally, flatten to get attention vector
    n['conv4-a'] = L.Flatten(name='conv4-a', bottom='conv4_weighted_b')

    # Fusion
    n['fc7-concat'] = L.Concat(name='fc7-concat', bottom=[fc7_out_name, 'conv4-a'])
    fc8_out_name = add_wrapped_fc_layer(n, 'fc8', 'fc7-concat', 4096, param=DEFAULT_DECAY_PARAM, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc8_out_name, param=DEFAULT_DECAY_PARAM)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()

'''
Weight map learned from keypoint class only
'''
def create_model_chcnn_kpc_only(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    train_data_lmdb_path = lmdb_paths[0]
    keypoint_class_lmdb_path = lmdb_paths[1]
    train_label_lmdb_path = lmdb_paths[2]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=train_data_lmdb_path, transform_param=data_transform_param)
    n['keypoint_class'] = L.Data(name='keypoint_class', batch_size=batch_size, backend=P.Data.LMDB, source=keypoint_class_lmdb_path)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=train_label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # Image (Render for CNN) features
    conv1_param = dict(num_output=96, kernel_size=11, stride=4)
    pool1_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2_param = dict(num_output=256, pad=2, kernel_size=5, group=2)
    pool2_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv3_param = dict(num_output=384, pad=1, kernel_size=3)
    conv4_param = dict(num_output=384, pad=1, kernel_size=3, group=2)
    conv5_param = dict(num_output=256, pad=1, kernel_size=3, group=2)
    pool5_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_param, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_param, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_param)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_param, param=DEFAULT_DECAY_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_param, param=DEFAULT_DECAY_PARAM, pooling_param=pool5_param)
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6', conv5_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7', fc6_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Keypoint class features
    fc_keypoint_class_name = add_wrapped_fc_layer(n, 'fc-keypoint-class', 'keypoint_class', 34, use_relu=False)

    # Attention map
    attn_param = dict(shape=dict(dim=[0, 1, 13, 13]))
    fc_keypoint_concat_name = add_wrapped_fc_layer(n, 'fc-keypoint-concat', fc_keypoint_class_name, 169, use_relu=False)
    n['fc-keypoint-concat-softmax'] = L.Softmax(name='fc-keypoint-concat-softmax', bottom=fc_keypoint_concat_name)
    n['attn'] = L.Reshape(name='attn', bottom='fc-keypoint-concat-softmax', reshape_param=attn_param)

    # Get conv4 attention vector. First, slice channels
    add_slice_layer(n, 'conv4', ['conv4_slice_' + str(i) for i in range(384)], dict(
        axis=1, slice_point=range(1, 384)
    ))
    # Then weigh each channel by the map
    for i in range(384):
        slice_bottom_name = 'conv4_slice_' + str(i)
        top_name = 'conv4_weighted_' + str(i)
        n[top_name] = L.Eltwise(name=top_name, bottom=[slice_bottom_name, 'attn'], eltwise_param=dict(operation=P.Eltwise.PROD))
    # Combine weighted maps
    n['conv4_weighted'] = L.Concat(name='conv4_weighted', bottom=['conv4_weighted_' + str(i) for i in range(384)])
    # Sum across width
    add_slice_layer(n, 'conv4_weighted', ['conv4_weighted_slice_' + str(i) for i in range(13)], dict(
        axis=3, slice_point=range(1, 13)
    ))
    n['conv4_weighted_a'] = L.Eltwise(name='conv4_weighted_a', bottom=['conv4_weighted_slice_' + str(i) for i in range(13)], eltwise_param=dict(operation=P.Eltwise.SUM))
    # Sum across height
    add_slice_layer(n, 'conv4_weighted_a', ['conv4_weighted_a_slice_' + str(i) for i in range(13)], dict(
        axis=2, slice_point=range(1, 13)
    ))
    n['conv4_weighted_b'] = L.Eltwise(name='conv4_weighted_a', bottom=['conv4_weighted_a_slice_' + str(i) for i in range(13)], eltwise_param=dict(operation=P.Eltwise.SUM))
    # Finally, flatten to get attention vector
    n['conv4-a'] = L.Flatten(name='conv4-a', bottom='conv4_weighted_b')

    # Fusion
    n['fc7-concat'] = L.Concat(name='fc7-concat', bottom=[fc7_out_name, 'conv4-a'])
    fc8_out_name = add_wrapped_fc_layer(n, 'fc8', 'fc7-concat', 4096, param=DEFAULT_DECAY_PARAM, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc8_out_name, param=DEFAULT_DECAY_PARAM)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()


def main(model_name, train_eval_with_pascal, initial_weights_path, perturb_sigma):
    '''
    Create an experiment to train the specified model configuration.
    :param model_name: The name of the model/network configuration to use
    :param train_eval_with_pascal: Bool indicating whether to train and evaluate on PASCAL 3D+ data
    :param initial_weights_path: Path to the initial weights to use
    :param perturb_sigma: The sigma of the Gaussian by which to perturb the keypoint map, if applicable
    :return:
    '''

    # Render for CNN
    if model_name == 'R4CNN':
        sub_lmdb_names = ['image_lmdb', 'viewpoint_label_lmdb']
        input_data_shapes = dict(data=(1, 3, 227, 227))
        create_model_fn = create_model_r4cnn
        pascal_test_only = False

    # Chessboard keypoint map
    elif model_name == 'CH-CNN':
        sub_lmdb_names = ['image_lmdb', 'chessboard_dt_map_lmdb', 'keypoint_class_lmdb', 'viewpoint_label_lmdb']
        input_data_shapes = dict(data=(1, 3, 227, 227), keypoint_map=(1, 1, 227, 227), keypoint_class=(1, 34, 1, 1))
        create_model_fn = create_model_chcnn_chessboard
        pascal_test_only = False

    # Manhattan keypoint map
    elif model_name == 'CH-CNN_manhattan':
        sub_lmdb_names = ['image_lmdb', 'manhattan_dt_map_lmdb', 'keypoint_class_lmdb', 'viewpoint_label_lmdb']
        input_data_shapes = dict(data=(1, 3, 227, 227), keypoint_map=(1, 1, 227, 227), keypoint_class=(1, 34, 1, 1))
        create_model_fn = create_model_chcnn_manhattan
        pascal_test_only = False

    # Euclidean keypoint map
    elif model_name == 'CH-CNN_euclidean':
        sub_lmdb_names = ['image_lmdb', 'euclidean_dt_map_lmdb', 'keypoint_class_lmdb', 'viewpoint_label_lmdb']
        input_data_shapes = dict(data=(1, 3, 227, 227), keypoint_map=(1, 1, 227, 227), keypoint_class=(1, 34, 1, 1))
        create_model_fn = create_model_chcnn_euclidean
        pascal_test_only = False

    # Gaussian keypoint map
    elif model_name == 'CH-CNN_gaussian':
        sub_lmdb_names = ['image_lmdb', 'gaussian_keypoint_map_lmdb', 'keypoint_class_lmdb', 'viewpoint_label_lmdb']
        input_data_shapes = dict(data=(1, 3, 227, 227), keypoint_map=(1, 1, 227, 227), keypoint_class=(1, 34, 1, 1))
        create_model_fn = create_model_chcnn_gaussian
        pascal_test_only = False

    # Fixed uniform weight map
    elif model_name == 'fixed_weight_map_uniform':
        sub_lmdb_names = ['image_lmdb', 'viewpoint_label_lmdb']
        input_data_shapes = dict(data=(1, 3, 227, 227))
        create_model_fn = create_model_uniform_weights
        pascal_test_only = False

    # Fixed Gaussian weight map
    elif model_name == 'fixed_weight_map_gaussian':
        sub_lmdb_names = ['image_lmdb', 'gaussian_attn_map_lmdb', 'viewpoint_label_lmdb']
        input_data_shapes = dict(data=(1, 3, 227, 227), attn=(1, 1, 13, 13))
        create_model_fn = create_model_gaussian_weights
        pascal_test_only = False

    # Just keypoint map
    elif model_name == 'CH-CNN_kpm_only':
        sub_lmdb_names = ['image_lmdb', 'chessboard_dt_map_lmdb', 'viewpoint_label_lmdb']
        input_data_shapes = dict(data=(1, 3, 227, 227), keypoint_map=(1, 1, 227, 227))
        create_model_fn = create_model_chcnn_kpm_only
        pascal_test_only = False

    # Just keypoint class
    elif model_name == 'CH-CNN_kpc_only':
        sub_lmdb_names = ['image_lmdb', 'keypoint_class_lmdb', 'viewpoint_label_lmdb']
        input_data_shapes = dict(data=(1, 3, 227, 227), keypoint_class=(1, 34, 1, 1))
        create_model_fn = create_model_chcnn_kpc_only
        pascal_test_only = False

    # Blank keypoint map (test only)
    elif model_name == 'CH-CNN_lost_kpm':
        sub_lmdb_names = ['image_lmdb', 'zero_keypoint_map_lmdb', 'keypoint_class_lmdb', 'viewpoint_label_lmdb']
        input_data_shapes = dict(data=(1, 3, 227, 227), keypoint_map=(1, 1, 227, 227), keypoint_class=(1, 34, 1, 1))
        create_model_fn = create_model_chcnn_chessboard
        pascal_test_only = True

    # Blank keypoint class (test only)
    elif model_name == 'CH-CNN_lost_kpc':
        sub_lmdb_names = ['image_lmdb', 'chessboard_dt_map_lmdb', 'zero_keypoint_class_lmdb', 'viewpoint_label_lmdb']
        input_data_shapes = dict(data=(1, 3, 227, 227), keypoint_map=(1, 1, 227, 227), keypoint_class=(1, 34, 1, 1))
        create_model_fn = create_model_chcnn_chessboard
        pascal_test_only = True

    # Blank keypoint map and class (test only)
    elif model_name == 'CH-CNN_lost_kpm_kpc':
        sub_lmdb_names = ['image_lmdb', 'zero_keypoint_map_lmdb', 'zero_keypoint_class_lmdb', 'viewpoint_label_lmdb']
        input_data_shapes = dict(data=(1, 3, 227, 227), keypoint_map=(1, 1, 227, 227), keypoint_class=(1, 34, 1, 1))
        create_model_fn = create_model_chcnn_chessboard
        pascal_test_only = True

    # Perturbed keypoint (test only)
    elif model_name == 'CH-CNN_perturb_kpm':
        sub_lmdb_names = ['image_lmdb', 'perturbed_%d_chessboard_dt_map_lmdb' % perturb_sigma, 'keypoint_class_lmdb', 'viewpoint_label_lmdb']
        input_data_shapes = dict(data=(1, 3, 227, 227), keypoint_map=(1, 1, 227, 227), keypoint_class=(1, 34, 1, 1))
        create_model_fn = create_model_chcnn_chessboard
        pascal_test_only = True

    # Force PASCAL 3D+ usage if the configuration only evaluates on PASCAL 3D+ test set
    if pascal_test_only:
        train_eval_with_pascal = True

    ### Define the training and validation/test sets ###
    if train_eval_with_pascal:
        train_lmdb_root = gv.g_corresp_pascal_train_lmdb_folder
        test_lmdb_root = gv.g_corresp_pascal_test_lmdb_folder
    else:
        train_lmdb_root = gv.g_corresp_syn_train_lmdb_folder
        test_lmdb_root = gv.g_corresp_syn_test_lmdb_folder

    # Set LMDB paths
    train_lmdb_paths = [os.path.join(train_lmdb_root, lmdb_name) for lmdb_name in sub_lmdb_names]
    test_lmdb_paths = [os.path.join(test_lmdb_root, lmdb_name) for lmdb_name in sub_lmdb_names]

    # Verify the models
    if not pascal_test_only:
        z_train_model = create_model_fn(train_lmdb_paths, BATCH_SIZE)
        verify_netspec(z_train_model, initial_weights_path)
    z_test_model = create_model_fn(test_lmdb_paths, BATCH_SIZE)
    verify_netspec(z_test_model, initial_weights_path)

    # Get experiment folder names
    exp_folder_names = os.listdir(gv.g_experiments_root_folder)
    # Get last experiment folder name
    if len(exp_folder_names) != 0:
        last_exp_folder_name = sorted(exp_folder_names)[-1]
        # Extract experiment number
        last_exp_num = int(last_exp_folder_name[:6])
        cur_exp_num = last_exp_num + 1
    else:
        cur_exp_num = 1

    # Get current date
    date_str = time.strftime('%m-%d-%Y_%H:%M:%S')
    # Combine current experiment number with time
    cur_exp_folder_name = '%06d_%s' % (cur_exp_num, date_str)

    # Create experiment folder and subfolders
    cur_exp_folder_path = os.path.join(gv.g_experiments_root_folder, cur_exp_folder_name)
    evaluation_path = os.path.join(cur_exp_folder_path, 'evaluation')
    model_path = os.path.join(cur_exp_folder_path, 'model')
    progress_path = os.path.join(cur_exp_folder_path, 'progress')
    snapshots_path = os.path.join(cur_exp_folder_path, 'snapshots')
    os.mkdir(cur_exp_folder_path)
    os.mkdir(evaluation_path)
    os.mkdir(model_path)
    os.mkdir(progress_path)
    os.mkdir(snapshots_path)

    # Create train and test model files
    if not pascal_test_only:
        with open(os.path.join(model_path, 'train.prototxt'), 'w') as f:
            f.write(str(z_train_model))
    with open(os.path.join(model_path, 'test.prototxt'), 'w') as f:
        f.write(str(z_test_model))
    # Create deploy model file
    deploy_model_str = netspec_to_deploy_prototxt_str(z_test_model, input_data_shapes)
    with open(os.path.join(model_path, 'deploy.prototxt'), 'w') as f:
        f.write(deploy_model_str)

    ### Save solver parameters ###
    if train_eval_with_pascal:
        solver_dict = merge_dicts(DEFAULT_SOLVER_DICT, dict(
            train_net=os.path.join(model_path, 'train.prototxt'),
            test_net=os.path.join(model_path, 'test.prototxt'),
            max_iter=6000,
            display=10,
            snapshot=200,
            test_interval=200
        ))
    else:
        solver_dict = merge_dicts(DEFAULT_SOLVER_DICT, dict(
            train_net=os.path.join(model_path, 'train.prototxt'),
            test_net=os.path.join(model_path, 'test.prototxt')
        ))

    solver_str = dict_to_solver_text(solver_dict)
    with open(os.path.join(model_path, 'solver.prototxt'), 'w') as f:
        f.write(solver_str)

    ### CREATE TRAINING SCRIPT ###
    with open(os.path.join(gv.g_corresp_model_root_folder, 'train_model.sh'), 'r') as f:
        train_script_contents = f.read()
    # Replace variables in script
    train_script_contents = train_script_contents.replace('[[INITIAL_WEIGHTS]]', initial_weights_path)
    train_script_contents = train_script_contents.replace('[[EXPERIMENT_FOLDER_NAME]]', cur_exp_folder_name)
    train_script_contents = train_script_contents.replace('[[EXPERIMENTS_ROOT]]', gv.g_experiments_root_folder)
    # Save script
    with open(os.path.join(model_path, 'train.sh'), 'w') as f:
        f.write(train_script_contents)
    # Make script executable
    os.chmod(os.path.join(model_path, 'train.sh'), 0744)

    ### CREATE RESUME SCRIPT ###
    with open(os.path.join(gv.g_corresp_model_root_folder, 'resume_model.sh'), 'r') as f:
        resume_script_contents = f.read()
    # Replace variables in script
    resume_script_contents = resume_script_contents.replace('[[EXPERIMENT_FOLDER_NAME]]', cur_exp_folder_name)
    resume_script_contents = resume_script_contents.replace('[[EXPERIMENTS_ROOT]]', gv.g_experiments_root_folder)
    # Save script
    with open(os.path.join(model_path, 'resume.sh'), 'w') as f:
        f.write(resume_script_contents)
    # Make script executable
    os.chmod(os.path.join(model_path, 'resume.sh'), 0744)

    # Add NOT_STARTED file
    with open(os.path.join(cur_exp_folder_path, 'NOT_STARTED'), 'w') as f:
        f.write('')

    # Add net visualization
    draw_net_script_path = os.path.join(gv.g_pycaffe_path, 'draw_net.py')
    model_prototxt_path = os.path.join(model_path, 'test.prototxt')
    visualization_path = os.path.join(cur_exp_folder_path, 'net.png')
    subprocess.call(['python', draw_net_script_path, model_prototxt_path, visualization_path, '--rankdir', 'TB'])

    # Add evaluation arguments file
    with open(os.path.join(evaluation_path, 'evalAcc_args.txt'), 'w') as f:
        # model_proto
        f.write(os.path.join(model_path, 'deploy.prototxt') + os.linesep)
        # model_weights (need to replace ### with iter number)
        f.write(os.path.join(snapshots_path, 'snapshot_iter_###.caffemodel') + os.linesep)
        # test_root
        f.write(test_lmdb_root + os.linesep)
        # output_keys (assumed to be three keys before loss and accuracy layers)
        layer_names = map(lambda x: x.name, z_test_model.layer)
        for output_layer_name in layer_names[-9:-6]:
            f.write(output_layer_name + os.linesep)
        # lmdb_info
        for i, lmdb_name in enumerate(sub_lmdb_names):
            # Assume the ith data layer uses the ith LMDB
            f.write(layer_names[i] + os.linesep)
            f.write(lmdb_name + os.linesep)
            f.write(('True' if lmdb_name == 'image_lmdb' else 'False') + os.linesep)

    ### CREATE README FILE ###
    with open(os.path.join(gv.g_corresp_model_root_folder, 'README.md.example'), 'r') as f:
        readme_contents = f.read()
    # Replace variables in script
    readme_contents = readme_contents.replace('[[EXPERIMENT_NUM]]', str(cur_exp_num))
    readme_contents = readme_contents.replace('[[MODEL_NAME]]', model_name)
    readme_contents = readme_contents.replace('[[INITIAL_WEIGHTS]]', initial_weights_path)
    readme_contents = readme_contents.replace('[[TRAIN_ROOT]]', train_lmdb_root)
    readme_contents = readme_contents.replace('[[TEST_ROOT]]', test_lmdb_root)
    # Populate other notes
    num_conseq_empty_lines = 0
    other_notes = ''
    print('Enter other notes (press enter 3 times in a row to exit):')
    # Enter notes with up to two returns between paragraphs
    while num_conseq_empty_lines < 2:
        line = raw_input()
        other_notes += '%s\n' % line
        num_conseq_empty_lines = (0 if line else num_conseq_empty_lines + 1)
    readme_contents = readme_contents.replace('[[OTHER_NOTES]]', other_notes.rstrip())

    # Save readme
    with open(os.path.join(cur_exp_folder_path, 'README.md'), 'w') as f:
        f.write(readme_contents)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='The name of the model/network')
    parser.add_argument('--pascal', action='store_true', help='Flag to train and evaluate on PASCAL 3D+')
    parser.add_argument('--init_weight_path', type=str, nargs=1, default=gv.g_render4cnn_weights_path, help='Path of weights to intialize the model with. It is set to the original R4CNN weights by default')
    parser.add_argument('--perturb_sigma', type=int, nargs=1, default=0, help='How much to perturb the keypoint map by. Only used for keypoint perturbation models')

    args = parser.parse_args()

    main(args.model_name, args.pascal, args.init_weight_path, args.perturb_sigma)