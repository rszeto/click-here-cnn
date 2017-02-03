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
HIGHER_LR_MULT_PARAM = [dict(lr_mult=10, decay_mult=1), dict(decay_mult=0)]
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
SYN_SOLVER_DICT = dict(
    train_net=None,
    test_net=None,
    test_iter=15,
    test_interval=2000,
    base_lr=0.0001,
    lr_policy='fixed',
    max_iter=500000,
    display=20,
    momentum=0.9,
    momentum2=0.999,
    weight_decay=0.0005,
    snapshot=2000,
    snapshot_prefix='snapshot',
    solver_mode='GPU',
    type='Adam'
)

REAL_VAL_SOLVER_DICT = dict(
    train_net=None,
    test_net=None,
    test_iter=11,
    test_interval=11,
    base_lr=0.00001,
    lr_policy='fixed',
    max_iter=10000,
    display=20,
    momentum=0.9,
    momentum2=0.999,
    weight_decay=0.0005,
    snapshot=11,
    snapshot_prefix='snapshot',
    solver_mode='GPU',
    type='Adam'
)

REAL_SOLVER_DICT = dict(
    train_net=None,
    test_net=None,
    test_iter=36,
    test_interval=36,
    base_lr=0.00001,
    lr_policy='fixed',
    max_iter=10000,
    display=20,
    momentum=0.9,
    momentum2=0.999,
    weight_decay=0.0005,
    snapshot=36,
    snapshot_prefix='real',
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
def verify_netspec(net_param):
    temp_file = tempfile.NamedTemporaryFile()
    # Write prototxt to file, and flush to make sure whole prototxt is written
    temp_file.write(str(net_param))
    temp_file.flush()
    # Try to make the net. Caffe will crash everything if it fails. This cannot be caught.
    caffe.Net(temp_file.name, caffe.TRAIN)

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

# Re-learn prediction layers, fixed conv
def create_model_r4cnn_fc_from_scratch(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
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
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_param, param=NO_LEARN_PARAM, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_param, param=NO_LEARN_PARAM, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_param, param=NO_LEARN_PARAM)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_param, param=NO_LEARN_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_param, param=NO_LEARN_PARAM, pooling_param=pool5_param)
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6-new', conv5_out_name, 4096, param=DEFAULT_DECAY_PARAM, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7-new', fc6_out_name, 4096, param=DEFAULT_DECAY_PARAM, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc7_out_name)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()


# Re-learn last two conv layers and FC layers, as per R4CNN paper
def create_model_r4cnn_orig_train_proc(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
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
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_param, param=NO_LEARN_PARAM, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_param, param=NO_LEARN_PARAM, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_param, param=NO_LEARN_PARAM)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_param, param=NO_LEARN_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_param, param=DEFAULT_DECAY_PARAM, pooling_param=pool5_param)
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6', conv5_out_name, 4096, param=DEFAULT_DECAY_PARAM, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7', fc6_out_name, 4096, param=DEFAULT_DECAY_PARAM, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc7_out_name)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()


def create_model_r4cnnp_fixed_conv(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    data_lmdb_path = lmdb_paths[0]
    label_lmdb_path = lmdb_paths[1]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=data_lmdb_path, transform_param=data_transform_param)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=label_lmdb_path)
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
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_param, param=NO_LEARN_PARAM, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_param, param=NO_LEARN_PARAM, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_param, param=NO_LEARN_PARAM)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_param, param=NO_LEARN_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_param, param=NO_LEARN_PARAM, pooling_param=pool5_param)
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6', conv5_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7', fc6_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc8_out_name = add_wrapped_fc_layer(n, 'fc8', fc7_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc8_out_name)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()

def create_model_r4cnnpp(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    data_lmdb_path = lmdb_paths[0]
    label_lmdb_path = lmdb_paths[1]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=data_lmdb_path, transform_param=data_transform_param)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=label_lmdb_path)
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
    fc8_out_name = add_wrapped_fc_layer(n, 'fc8', fc7_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc9_out_name = add_wrapped_fc_layer(n, 'fc9', fc8_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc9_out_name)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()

def create_model_r(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    train_data_lmdb_path = lmdb_paths[0]
    pool5_weight_maps_lmdb_path = lmdb_paths[1]
    train_label_lmdb_path = lmdb_paths[2]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=train_data_lmdb_path, transform_param=data_transform_param)
    n['pool5_weight_map'] = L.Data(name='pool5_weight_map', batch_size=batch_size, backend=P.Data.LMDB, source=pool5_weight_maps_lmdb_path)
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
    pool5_weighted_param = dict(operation=P.Eltwise.PROD)
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_param, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_param, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_param)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_param, param=DEFAULT_DECAY_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_param, param=DEFAULT_DECAY_PARAM, pooling_param=pool5_param)

    # Get pool5 weighted features. First, slice channels
    add_slice_layer(n, 'pool5', ['pool5_slice_' + str(i) for i in range(256)], dict(
        axis=1, slice_point=range(1, 256)
    ))
    # Then weigh each channel by the map
    for i in range(256):
        slice_bottom_name = 'pool5_slice_' + str(i)
        top_name = 'pool5_weighted_' + str(i)
        n[top_name] = L.Eltwise(name=top_name, bottom=[slice_bottom_name, 'pool5_weight_map'], eltwise_param=pool5_weighted_param)
    # Finally, combine weighted maps
    n['pool5_weighted'] = L.Concat(name='pool5_weighted', bottom=['pool5_weighted_' + str(i) for i in range(256)])

    # FC features
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6_new', 'pool5_weighted', 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7_new', fc6_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc7_out_name)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()

def create_model_s(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    train_data_lmdb_path = lmdb_paths[0]
    pool5_weight_maps_lmdb_path = lmdb_paths[1]
    train_label_lmdb_path = lmdb_paths[2]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=train_data_lmdb_path, transform_param=data_transform_param)
    n['pool5_weight_map'] = L.Data(name='pool5_weight_map', batch_size=batch_size, backend=P.Data.LMDB, source=pool5_weight_maps_lmdb_path)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=train_label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # Parameters
    conv1_param = dict(num_output=96, kernel_size=11, stride=4)
    pool1_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2_param = dict(num_output=256, pad=2, kernel_size=5, group=2)
    pool2_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv3_param = dict(num_output=384, pad=1, kernel_size=3)
    conv4_param = dict(num_output=384, pad=1, kernel_size=3, group=2)
    conv5_param = dict(num_output=256, pad=1, kernel_size=3, group=2)
    pool5_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    pool5_weighted_param = dict(operation=P.Eltwise.PROD)
    fc7_sum_param = dict(operation=P.Eltwise.SUM)

    # Image (Render for CNN) features
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_param, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_param, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_param)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_param, param=DEFAULT_DECAY_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_param, param=DEFAULT_DECAY_PARAM, pooling_param=pool5_param)
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6', conv5_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7', fc6_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Get pool5 weighted features. First, slice channels
    add_slice_layer(n, 'pool5', ['pool5_slice_' + str(i) for i in range(256)], dict(
        axis=1, slice_point=range(1, 256)
    ))
    # Then weigh each channel by the map
    for i in range(256):
        slice_bottom_name = 'pool5_slice_' + str(i)
        top_name = 'pool5_weighted_' + str(i)
        n[top_name] = L.Eltwise(name=top_name, bottom=[slice_bottom_name, 'pool5_weight_map'], eltwise_param=pool5_weighted_param)
    # Finally, combine weighted maps
    n['pool5_weighted'] = L.Concat(name='pool5_weighted', bottom=['pool5_weighted_' + str(i) for i in range(256)])
    # Add FC features from weighted features
    fc6_a_out_name = add_wrapped_fc_layer(n, 'fc6-a', 'pool5_weighted', 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_a_out_name = add_wrapped_fc_layer(n, 'fc7-a', fc6_a_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Combined FC7 features
    n['fc7-sum'] = L.Eltwise(name='fc7-sum', bottom=[fc7_out_name, fc7_a_out_name], eltwise_param=fc7_sum_param)
    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', 'fc7-sum')
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()


'''
LRN after pool5_weighted. High learning rate for fc6/7-a and prediction layers. Fixed conv1-3 layers.
'''
def create_model_s2(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    train_data_lmdb_path = lmdb_paths[0]
    pool5_weight_maps_lmdb_path = lmdb_paths[1]
    train_label_lmdb_path = lmdb_paths[2]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=train_data_lmdb_path, transform_param=data_transform_param)
    n['pool5_weight_map'] = L.Data(name='pool5_weight_map', batch_size=batch_size, backend=P.Data.LMDB, source=pool5_weight_maps_lmdb_path)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=train_label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # Parameters
    conv1_conv_param = dict(num_output=96, kernel_size=11, stride=4)
    pool1_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2_conv_param = dict(num_output=256, pad=2, kernel_size=5, group=2)
    pool2_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv3_conv_param = dict(num_output=384, pad=1, kernel_size=3)
    conv4_conv_param = dict(num_output=384, pad=1, kernel_size=3, group=2)
    conv5_conv_param = dict(num_output=256, pad=1, kernel_size=3, group=2)
    pool5_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    pool5_weighted_param = dict(operation=P.Eltwise.PROD)
    fc7_sum_param = dict(operation=P.Eltwise.PROD)

    # Image (Render for CNN) features
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_conv_param, param=NO_LEARN_PARAM, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_conv_param, param=NO_LEARN_PARAM, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_conv_param, param=NO_LEARN_PARAM)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_conv_param, param=DEFAULT_DECAY_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_conv_param, param=DEFAULT_DECAY_PARAM, pooling_param=pool5_param)
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6', conv5_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7', fc6_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Get pool5 weighted features. First, slice channels
    add_slice_layer(n, 'pool5', ['pool5_slice_' + str(i) for i in range(256)], dict(
        axis=1, slice_point=range(1, 256)
    ))
    # Then weigh each channel by the map
    for i in range(256):
        slice_bottom_name = 'pool5_slice_' + str(i)
        top_name = 'pool5_weighted_' + str(i)
        n[top_name] = L.Eltwise(name=top_name, bottom=[slice_bottom_name, 'pool5_weight_map'], eltwise_param=pool5_weighted_param)
    # Finally, combine weighted maps
    n['pool5_weighted'] = L.Concat(name='pool5_weighted', bottom=['pool5_weighted_' + str(i) for i in range(256)])
    # Add LRN and FC features from weighted features
    n['pool5_weighted_lrn'] = lrn('pool5_weighted_lrn', 'pool5_weighted', DEFAULT_LRN_PARAM)
    fc6_a_out_name = add_wrapped_fc_layer(n, 'fc6-a', 'pool5_weighted_lrn', 4096, param=HIGHER_LR_MULT_PARAM, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_a_out_name = add_wrapped_fc_layer(n, 'fc7-a', fc6_a_out_name, 4096, param=HIGHER_LR_MULT_PARAM, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Combined FC7 features
    n['fc7-sum'] = L.Eltwise(name='fc7-sum', bottom=[fc7_out_name, fc7_a_out_name], eltwise_param=fc7_sum_param)
    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', 'fc7-sum', param=HIGHER_LR_MULT_PARAM)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()


def create_model_s_fixed_conv_lrn(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    train_data_lmdb_path = lmdb_paths[0]
    pool5_weight_maps_lmdb_path = lmdb_paths[1]
    train_label_lmdb_path = lmdb_paths[2]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=train_data_lmdb_path, transform_param=data_transform_param)
    n['pool5_weight_map'] = L.Data(name='pool5_weight_map', batch_size=batch_size, backend=P.Data.LMDB, source=pool5_weight_maps_lmdb_path)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=train_label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # Parameters
    conv1_conv_param = dict(num_output=96, kernel_size=11, stride=4)
    pool1_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2_conv_param = dict(num_output=256, pad=2, kernel_size=5, group=2)
    pool2_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv3_conv_param = dict(num_output=384, pad=1, kernel_size=3)
    conv4_conv_param = dict(num_output=384, pad=1, kernel_size=3, group=2)
    conv5_conv_param = dict(num_output=256, pad=1, kernel_size=3, group=2)
    pool5_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    pool5_weighted_param = dict(operation=P.Eltwise.PROD)
    fc7_sum_param = dict(operation=P.Eltwise.SUM)

    # Image (Render for CNN) features
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_conv_param, param=NO_LEARN_PARAM, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_conv_param, param=NO_LEARN_PARAM, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_conv_param, param=NO_LEARN_PARAM)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_conv_param, param=NO_LEARN_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_conv_param, param=NO_LEARN_PARAM, pooling_param=pool5_param)
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6', conv5_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7', fc6_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Get pool5 weighted features. First, slice channels
    add_slice_layer(n, 'pool5', ['pool5_slice_' + str(i) for i in range(256)], dict(
        axis=1, slice_point=range(1, 256)
    ))
    # Then weigh each channel by the map
    for i in range(256):
        slice_bottom_name = 'pool5_slice_' + str(i)
        top_name = 'pool5_weighted_' + str(i)
        n[top_name] = L.Eltwise(name=top_name, bottom=[slice_bottom_name, 'pool5_weight_map'], eltwise_param=pool5_weighted_param)
    # Finally, combine weighted maps
    n['pool5_weighted'] = L.Concat(name='pool5_weighted', bottom=['pool5_weighted_' + str(i) for i in range(256)])
    n['pool5_weighted_lrn'] = lrn('pool5_weighted_lrn', 'pool5_weighted', DEFAULT_LRN_PARAM)
    # Add FC features from weighted features
    fc6_a_out_name = add_wrapped_fc_layer(n, 'fc6-a', 'pool5_weighted_lrn', 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_a_out_name = add_wrapped_fc_layer(n, 'fc7-a', fc6_a_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Combined FC7 features
    n['fc7-sum'] = L.Eltwise(name='fc7-sum', bottom=[fc7_out_name, fc7_a_out_name], eltwise_param=fc7_sum_param)
    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', 'fc7-sum')
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()


# Model s with all FC layers trained from scratch
def create_model_s_fixed_conv_scratch_fc(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    train_data_lmdb_path = lmdb_paths[0]
    pool5_weight_maps_lmdb_path = lmdb_paths[1]
    train_label_lmdb_path = lmdb_paths[2]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=train_data_lmdb_path, transform_param=data_transform_param)
    n['pool5_weight_map'] = L.Data(name='pool5_weight_map', batch_size=batch_size, backend=P.Data.LMDB, source=pool5_weight_maps_lmdb_path)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=train_label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # Parameters
    conv1_conv_param = dict(num_output=96, kernel_size=11, stride=4)
    pool1_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2_conv_param = dict(num_output=256, pad=2, kernel_size=5, group=2)
    pool2_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv3_conv_param = dict(num_output=384, pad=1, kernel_size=3)
    conv4_conv_param = dict(num_output=384, pad=1, kernel_size=3, group=2)
    conv5_conv_param = dict(num_output=256, pad=1, kernel_size=3, group=2)
    pool5_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    pool5_weighted_param = dict(operation=P.Eltwise.PROD)
    fc7_sum_param = dict(operation=P.Eltwise.SUM)

    # Image (Render for CNN) features
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_conv_param, param=NO_LEARN_PARAM, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_conv_param, param=NO_LEARN_PARAM, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_conv_param, param=NO_LEARN_PARAM)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_conv_param, param=NO_LEARN_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_conv_param, param=NO_LEARN_PARAM, pooling_param=pool5_param)
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6-new', conv5_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7-new', fc6_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Get pool5 weighted features. First, slice channels
    add_slice_layer(n, 'pool5', ['pool5_slice_' + str(i) for i in range(256)], dict(
        axis=1, slice_point=range(1, 256)
    ))
    # Then weigh each channel by the map
    for i in range(256):
        slice_bottom_name = 'pool5_slice_' + str(i)
        top_name = 'pool5_weighted_' + str(i)
        n[top_name] = L.Eltwise(name=top_name, bottom=[slice_bottom_name, 'pool5_weight_map'], eltwise_param=pool5_weighted_param)
    # Finally, combine weighted maps
    n['pool5_weighted'] = L.Concat(name='pool5_weighted', bottom=['pool5_weighted_' + str(i) for i in range(256)])
    # Add FC features from weighted features
    fc6_a_out_name = add_wrapped_fc_layer(n, 'fc6-a', 'pool5_weighted', 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_a_out_name = add_wrapped_fc_layer(n, 'fc7-a', fc6_a_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Combined FC7 features
    n['fc7-sum'] = L.Eltwise(name='fc7-sum', bottom=[fc7_out_name, fc7_a_out_name], eltwise_param=fc7_sum_param)
    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', 'fc7-sum')
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()


def create_model_t_fixed_conv(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    train_data_lmdb_path = lmdb_paths[0]
    pool5_weight_maps_lmdb_path = lmdb_paths[1]
    train_label_lmdb_path = lmdb_paths[2]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=train_data_lmdb_path, transform_param=data_transform_param)
    n['pool5_weight_map'] = L.Data(name='pool5_weight_map', batch_size=batch_size, backend=P.Data.LMDB, source=pool5_weight_maps_lmdb_path)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=train_label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # Parameters
    conv1_conv_param = dict(num_output=96, kernel_size=11, stride=4)
    pool1_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2_conv_param = dict(num_output=256, pad=2, kernel_size=5, group=2)
    pool2_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv3_conv_param = dict(num_output=384, pad=1, kernel_size=3)
    conv4_conv_param = dict(num_output=384, pad=1, kernel_size=3, group=2)
    conv5_conv_param = dict(num_output=256, pad=1, kernel_size=3, group=2)
    pool5_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    pool5_weighted_param = dict(operation=P.Eltwise.PROD)
    fc6_sum_param = dict(operation=P.Eltwise.MAX)

    # Image (Render for CNN) features
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_conv_param, param=NO_LEARN_PARAM, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_conv_param, param=NO_LEARN_PARAM, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_conv_param, param=NO_LEARN_PARAM)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_conv_param, param=NO_LEARN_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_conv_param, param=NO_LEARN_PARAM, pooling_param=pool5_param)
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6', conv5_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Get pool5 weighted features. First, slice channels
    add_slice_layer(n, 'pool5', ['pool5_slice_' + str(i) for i in range(256)], dict(
        axis=1, slice_point=range(1, 256)
    ))
    # Then weigh each channel by the map
    for i in range(256):
        slice_bottom_name = 'pool5_slice_' + str(i)
        top_name = 'pool5_weighted_' + str(i)
        n[top_name] = L.Eltwise(name=top_name, bottom=[slice_bottom_name, 'pool5_weight_map'], eltwise_param=pool5_weighted_param)
    # Finally, combine weighted maps
    n['pool5_weighted'] = L.Concat(name='pool5_weighted', bottom=['pool5_weighted_' + str(i) for i in range(256)])
    # Add FC features from weighted features
    fc6_a_out_name = add_wrapped_fc_layer(n, 'fc6-a', 'pool5_weighted', 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Combined FC6 and FC7 features
    n['fc6-sum'] = L.Eltwise(name='fc6-sum', bottom=[fc6_out_name, fc6_a_out_name], eltwise_param=fc6_sum_param)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7-new', 'fc6-sum', 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc7_out_name)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()


def create_model_u_fixed_conv(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    train_data_lmdb_path = lmdb_paths[0]
    pool5_weight_maps_lmdb_path = lmdb_paths[1]
    train_label_lmdb_path = lmdb_paths[2]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=train_data_lmdb_path, transform_param=data_transform_param)
    n['pool5_weight_map'] = L.Data(name='pool5_weight_map', batch_size=batch_size, backend=P.Data.LMDB, source=pool5_weight_maps_lmdb_path)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=train_label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # Parameters
    conv1_conv_param = dict(num_output=96, kernel_size=11, stride=4)
    pool1_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2_conv_param = dict(num_output=256, pad=2, kernel_size=5, group=2)
    pool2_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv3_conv_param = dict(num_output=384, pad=1, kernel_size=3)
    conv4_conv_param = dict(num_output=384, pad=1, kernel_size=3, group=2)
    conv5_conv_param = dict(num_output=256, pad=1, kernel_size=3, group=2)
    pool5_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    pool5_weighted_param = dict(operation=P.Eltwise.PROD)

    # Image (Render for CNN) features
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_conv_param, param=NO_LEARN_PARAM, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_conv_param, param=NO_LEARN_PARAM, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_conv_param, param=NO_LEARN_PARAM)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_conv_param, param=NO_LEARN_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_conv_param, param=NO_LEARN_PARAM, pooling_param=pool5_param)
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6', conv5_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7', fc6_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Get pool5 weighted features. First, slice channels
    add_slice_layer(n, 'pool5', ['pool5_slice_' + str(i) for i in range(256)], dict(
        axis=1, slice_point=range(1, 256)
    ))
    # Then weigh each channel by the map
    for i in range(256):
        slice_bottom_name = 'pool5_slice_' + str(i)
        top_name = 'pool5_weighted_' + str(i)
        n[top_name] = L.Eltwise(name=top_name, bottom=[slice_bottom_name, 'pool5_weight_map'], eltwise_param=pool5_weighted_param)
    # Finally, combine weighted maps
    n['pool5_weighted'] = L.Concat(name='pool5_weighted', bottom=['pool5_weighted_' + str(i) for i in range(256)])
    # Add FC features from weighted features
    fc6_a_out_name = add_wrapped_fc_layer(n, 'fc6-a', 'pool5_weighted', 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_a_out_name = add_wrapped_fc_layer(n, 'fc7-a', fc6_a_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Combined FC7 features
    n['fc7-concat'] = L.Concat(name='fc7-concat', bottom=[fc7_out_name, fc7_a_out_name])
    fc8_out_name = add_wrapped_fc_layer(n, 'fc8', 'fc7-concat', 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc8_out_name)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()


'''
LRN after pool5_weighted. High learning rate for fc6/7-a, fc8, and prediction layers. Fixed conv1-3 layers.
'''
def create_model_v(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    train_data_lmdb_path = lmdb_paths[0]
    pool5_weight_maps_lmdb_path = lmdb_paths[1]
    train_label_lmdb_path = lmdb_paths[2]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=train_data_lmdb_path, transform_param=data_transform_param)
    n['pool5_weight_map'] = L.Data(name='pool5_weight_map', batch_size=batch_size, backend=P.Data.LMDB, source=pool5_weight_maps_lmdb_path)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=train_label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # Parameters
    conv1_conv_param = dict(num_output=96, kernel_size=11, stride=4)
    pool1_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2_conv_param = dict(num_output=256, pad=2, kernel_size=5, group=2)
    pool2_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv3_conv_param = dict(num_output=384, pad=1, kernel_size=3)
    conv4_conv_param = dict(num_output=384, pad=1, kernel_size=3, group=2)
    conv5_conv_param = dict(num_output=256, pad=1, kernel_size=3, group=2)
    pool5_param = dict(pool=P.Pooling.MAX, kernel_size=3, stride=2)
    pool5_weighted_param = dict(operation=P.Eltwise.PROD)
    eltwise_sum_param = dict(operation=P.Eltwise.SUM)

    # Image (Render for CNN) features
    conv1_out_name = add_wrapped_conv_layer(n, 'conv1', 'data', conv1_conv_param, param=NO_LEARN_PARAM, pooling_param=pool1_param, lrn_param=DEFAULT_LRN_PARAM)
    conv2_out_name = add_wrapped_conv_layer(n, 'conv2', conv1_out_name, conv2_conv_param, param=NO_LEARN_PARAM, pooling_param=pool2_param, lrn_param=DEFAULT_LRN_PARAM)
    conv3_out_name = add_wrapped_conv_layer(n, 'conv3', conv2_out_name, conv3_conv_param, param=NO_LEARN_PARAM)
    conv4_out_name = add_wrapped_conv_layer(n, 'conv4', conv3_out_name, conv4_conv_param, param=DEFAULT_DECAY_PARAM)
    conv5_out_name = add_wrapped_conv_layer(n, 'conv5', conv4_out_name, conv5_conv_param, param=DEFAULT_DECAY_PARAM, pooling_param=pool5_param)
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6', conv5_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7', fc6_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Get pool5 attention vector. First, slice channels
    add_slice_layer(n, 'pool5', ['pool5_slice_' + str(i) for i in range(256)], dict(
        axis=1, slice_point=range(1, 256)
    ))
    # Then weigh each channel by the map
    for i in range(256):
        slice_bottom_name = 'pool5_slice_' + str(i)
        top_name = 'pool5_weighted_' + str(i)
        n[top_name] = L.Eltwise(name=top_name, bottom=[slice_bottom_name, 'pool5_weight_map'], eltwise_param=pool5_weighted_param)
    # Combine weighted maps
    n['pool5_weighted'] = L.Concat(name='pool5_weighted', bottom=['pool5_weighted_' + str(i) for i in range(256)])
    n['pool5_weighted_lrn'] = lrn('pool5_weighted_lrn', 'pool5_weighted', DEFAULT_LRN_PARAM)
    # Sum across width
    add_slice_layer(n, 'pool5_weighted_lrn', ['pool5_weighted_slice_' + str(i) for i in range(6)], dict(
        axis=3, slice_point=range(1, 6)
    ))
    n['pool5_weighted_a'] = L.Eltwise(name='pool5_weighted_a', bottom=['pool5_weighted_slice_' + str(i) for i in range(6)], eltwise_param=eltwise_sum_param)
    # Sum across height
    add_slice_layer(n, 'pool5_weighted_a', ['pool5_weighted_a_slice_' + str(i) for i in range(6)], dict(
        axis=2, slice_point=range(1, 6)
    ))
    n['pool5_weighted_b'] = L.Eltwise(name='pool5_weighted_a', bottom=['pool5_weighted_a_slice_' + str(i) for i in range(6)], eltwise_param=eltwise_sum_param)
    # Finally, flatten to get attention vector
    n['pool5-a'] = L.Flatten(name='pool5-a', bottom='pool5_weighted_b')

    # Concatenate fc7 with attention vector
    n['fc7-concat'] = L.Concat(name='fc7-concat', bottom=[fc7_out_name, 'pool5-a'])
    fc8_out_name = add_wrapped_fc_layer(n, 'fc8', 'fc7-concat', 4096, param=HIGHER_LR_MULT_PARAM, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc8_out_name, param=HIGHER_LR_MULT_PARAM)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()


def create_model_w(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
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
    pool_keypoint_map_param = dict(pool=P.Pooling.MAX, kernel_size=5, stride=5, pad=3)
    n['pool_keypoint_map'] = L.Pooling(name='pool_keypoint_map', bottom='keypoint_map', pooling_param=pool_keypoint_map_param)
    # n['flatten'] = L.Flatten(name='flatten', bottom='pool_keypoint_map')
    fc_keypoint_map_name = add_wrapped_fc_layer(n, 'fc_keypoint_map', 'pool_keypoint_map', 2116, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Fusion
    n['fc7-concat'] = L.Concat(name='fc7-concat', bottom=[fc7_out_name, fc_keypoint_map_name])
    fc8_out_name = add_wrapped_fc_layer(n, 'fc8', 'fc7-concat', 4096, param=HIGHER_LR_MULT_PARAM, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc8_out_name, param=HIGHER_LR_MULT_PARAM)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()

def main():
    # Potential arguments
    # sub_lmdb_names = ['image_lmdb', 'viewpoint_label_lmdb']
    # input_data_shapes = dict(data=(1, 3, 227, 227))
    # model_id = 'r4cnn_fc_from_scratch'
    # create_model_fn = create_model_r4cnn_fc_from_scratch

    # sub_lmdb_names = ['image_lmdb', 'viewpoint_label_lmdb']
    # input_data_shapes = dict(data=(1, 3, 227, 227))
    # create_model_fn = create_model_r4cnn_orig_train_proc

    # sub_lmdb_names = ['image_lmdb', 'viewpoint_label_lmdb']
    # input_data_shapes = dict(data=(1, 3, 227, 227))
    # create_model_fn = create_model_r4cnnp_fixed_conv

    # sub_lmdb_names = ['image_lmdb', 'pool5_weight_maps_lmdb', 'viewpoint_label_lmdb']
    # input_data_shapes = dict(data=(1, 3, 227, 227), pool5_weight_map=(1, 1, 6, 6))
    # create_model_fn = create_model_r

    # sub_lmdb_names = ['image_lmdb', 'pool5_weight_maps_lmdb', 'viewpoint_label_lmdb']
    # input_data_shapes = dict(data=(1, 3, 227, 227), pool5_weight_map=(1, 1, 6, 6))
    # create_model_fn = create_model_s

    # sub_lmdb_names = ['image_lmdb', 'pool5_weight_maps_lmdb', 'viewpoint_label_lmdb']
    # input_data_shapes = dict(data=(1, 3, 227, 227), pool5_weight_map=(1, 1, 6, 6))
    # model_id = 's2_prod'
    # create_model_fn = create_model_s2

    # sub_lmdb_names = ['image_lmdb', 'pool5_weight_maps_lmdb', 'viewpoint_label_lmdb']
    # input_data_shapes = dict(data=(1, 3, 227, 227), pool5_weight_map=(1, 1, 6, 6))
    # model_id = 's_fast-fc-a'
    # create_model_fn = create_model_s_fixed_conv

    # sub_lmdb_names = ['image_lmdb', 'pool5_weight_maps_lmdb', 'viewpoint_label_lmdb']
    # input_data_shapes = dict(data=(1, 3, 227, 227), pool5_weight_map=(1, 1, 6, 6))
    # model_id = 's_fixed_conv_lrn'
    # create_model_fn = create_model_s_fixed_conv_lrn

    # sub_lmdb_names = ['image_lmdb', 'pool5_weight_maps_lmdb', 'viewpoint_label_lmdb']
    # input_data_shapes = dict(data=(1, 3, 227, 227), pool5_weight_map=(1, 1, 6, 6))
    # model_id = 's_sum_fc-only_fc-scratch'
    # create_model_fn = create_model_s_fixed_conv_scratch_fc

    # sub_lmdb_names = ['image_lmdb', 'pool5_weight_maps_lmdb', 'viewpoint_label_lmdb']
    # input_data_shapes = dict(data=(1, 3, 227, 227), pool5_weight_map=(1, 1, 6, 6))
    # model_id = 't_max_fc-only'
    # create_model_fn = create_model_t_fixed_conv

    # sub_lmdb_names = ['image_lmdb', 'pool5_weight_maps_lmdb', 'viewpoint_label_lmdb']
    # input_data_shapes = dict(data=(1, 3, 227, 227), pool5_weight_map=(1, 1, 6, 6))
    # model_id = 'u_fc-only'
    # create_model_fn = create_model_u_fixed_conv

    sub_lmdb_names = ['image_lmdb', 'pool5_weight_maps_lmdb', 'viewpoint_label_lmdb']
    input_data_shapes = dict(data=(1, 3, 227, 227), pool5_weight_map=(1, 1, 6, 6))
    create_model_fn = create_model_v

    # Define the model
    # sub_lmdb_names = ['image_lmdb', 'euclidean_dt_map_lmdb', 'viewpoint_label_lmdb']
    # input_data_shapes = dict(data=(1, 3, 227, 227))
    # create_model_fn = create_model_w

    # Define initial weights
    initial_weights_path = gv.g_render4cnn_weights_path

    # Define the training and test sets
    # z_train_lmdb_root = gv.g_z_corresp_syn_train_lmdb_folder
    # z_test_lmdb_root = gv.g_z_corresp_syn_val_lmdb_folder
    # scratch_train_lmdb_root = gv.g_scratch_corresp_syn_train_lmdb_folder
    # scratch_test_lmdb_root = gv.g_scratch_corresp_syn_val_lmdb_folder
    z_train_lmdb_root = '/z/home/szetor/Documents/DENSO_VCA/RenderForCNN/data/lmdb/syn'
    z_test_lmdb_root = '/z/home/szetor/Documents/DENSO_VCA/RenderForCNN/data/lmdb/real/test'
    scratch_train_lmdb_root = z_train_lmdb_root.replace('/z/', '/scratch/')
    scratch_test_lmdb_root = z_test_lmdb_root.replace('/z/', '/scratch/')

    # Get model ID by stripping 'create_model_' from the function name
    model_id = create_model_fn.__name__.replace('create_model_', '')
    # Set /z LMDB paths
    z_train_lmdb_paths = [os.path.join(z_train_lmdb_root, lmdb_name) for lmdb_name in sub_lmdb_names]
    z_test_lmdb_paths = [os.path.join(z_test_lmdb_root, lmdb_name) for lmdb_name in sub_lmdb_names]
    # Set /scratch LMDB paths
    scratch_train_lmdb_paths = [os.path.join(scratch_train_lmdb_root, lmdb_name) for lmdb_name in sub_lmdb_names]
    scratch_test_lmdb_paths = [os.path.join(scratch_test_lmdb_root, lmdb_name) for lmdb_name in sub_lmdb_names]

    # # Verify the models using the LMDBs on /z
    # z_train_model = create_model_fn(z_train_lmdb_paths, BATCH_SIZE)
    # verify_netspec(z_train_model)
    # z_test_model = create_model_fn(z_test_lmdb_paths, BATCH_SIZE)
    # verify_netspec(z_test_model)

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
    scratch_train_model = create_model_fn(scratch_train_lmdb_paths, BATCH_SIZE)
    scratch_test_model = create_model_fn(scratch_test_lmdb_paths, BATCH_SIZE)
    with open(os.path.join(model_path, 'train.prototxt'), 'w') as f:
        f.write(str(scratch_train_model))
    with open(os.path.join(model_path, 'test.prototxt'), 'w') as f:
        f.write(str(scratch_test_model))
    # Create deploy model file
    deploy_model_str = netspec_to_deploy_prototxt_str(scratch_train_model, input_data_shapes)
    with open(os.path.join(model_path, 'deploy.prototxt'), 'w') as f:
        f.write(deploy_model_str)

    # Save solver parameters
    solver_override_params = dict(
        train_net=os.path.join(model_path, 'train.prototxt'),
        test_net=os.path.join(model_path, 'test.prototxt')
    )
    solver_params = merge_dicts(SYN_SOLVER_DICT, solver_override_params)
    solver_str = dict_to_solver_text(solver_params)
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

    # Add NOT_STARTED file
    with open(os.path.join(cur_exp_folder_path, 'NOT_STARTED'), 'w') as f:
        f.write('')

    # Add net visualization
    draw_net_script_path = os.path.join(gv.g_pycaffe_path, 'draw_net.py')
    model_prototxt_path = os.path.join(model_path, 'train.prototxt')
    visualization_path = os.path.join(cur_exp_folder_path, 'net.png')
    subprocess.call(['python', draw_net_script_path, model_prototxt_path, visualization_path, '--rankdir', 'TB'])

    # Add evaluation arguments file
    with open(os.path.join(evaluation_path, 'evalAcc_args.txt'), 'w') as f:
        # model_proto
        f.write(os.path.join(model_path, 'deploy.prototxt') + os.linesep)
        # model_weights (need to replace ### with iter number)
        f.write(os.path.join(snapshots_path, 'snapshot_iter_###.caffemodel') + os.linesep)
        # test_root
        f.write(z_test_lmdb_root + os.linesep)
        # output_keys (assumed to be three keys before loss and accuracy layers)
        layer_names = map(lambda x: x.name, scratch_train_model.layer)        
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
    readme_contents = readme_contents.replace('[[MODEL_NAME]]', model_id)
    readme_contents = readme_contents.replace('[[TRAIN_ROOT]]', scratch_train_lmdb_root)
    readme_contents = readme_contents.replace('[[TEST_ROOT]]', scratch_test_lmdb_root)
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
    main()