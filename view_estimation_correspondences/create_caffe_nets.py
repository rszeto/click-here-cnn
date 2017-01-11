import os
import sys
from warnings import warn
import tempfile
import shutil
from collections import OrderedDict

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
HIGHER_LR_MULT_PARAM = [dict(lr_mult=10, decay_mult=1), dict(decay_mult=0)]
DEFAULT_WEIGHT_FILLER = dict(type='gaussian', std=0.01)
DEFAULT_BIAS_FILLER = dict(type='constant', value=0)
DEFAULT_DROPOUT_RATIO = 0.5
DEFAULT_ANGLE_NAMES = ['azimuth', 'elevation', 'tilt']
DEFAULT_LOSS_WEIGHTS = [1, 1, 1]

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
    lr_policy='step',
    gamma=0.1,
    stepsize=100000,
    max_iter=100000,
    display=20,
    momentum=0.9,
    weight_decay=0.0005,
    snapshot=1000,
    snapshot_prefix=None,
    solver_mode='GPU'
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
WARNING: This modifies the net_param argument. Only use this after the given NetParameter has been saved to disk.
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
    # Delete data, softmax, and accuracy layers
    net_param = net_param
    for layer_type in ['Data', 'Silence', 'SoftmaxWithViewLoss', 'AccuracyView']:
        delete_layer_by_type(net_param, layer_type)
    # Delete label slice layer
    delete_layer_by_name(net_param, 'labe-slice')
    ret += str(net_param)
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

def create_model_j(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
    data_lmdb_path = lmdb_paths[0]
    data_keypoint_image_lmdb_path = lmdb_paths[1]
    data_keypoint_class_lmdb_path = lmdb_paths[2]
    label_lmdb_path = lmdb_paths[3]
    data_transform_param = dict(
        crop_size=crop_size,
        mean_file=imagenet_mean_file,
        mirror=False
    )

    n = caffe.NetSpec()
    # Data layers
    n['data'] = L.Data(name='data', batch_size=batch_size, backend=P.Data.LMDB, source=data_lmdb_path, transform_param=data_transform_param)
    n['data_keypoint_image'] = L.Data(name='data_keypoint_image', batch_size=batch_size, backend=P.Data.LMDB, source=data_keypoint_image_lmdb_path)
    n['data_keypoint_class'] = L.Data(name='data_keypoint_class', batch_size=batch_size, backend=P.Data.LMDB, source=data_keypoint_class_lmdb_path)
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

    # Keypoint image features
    n['pool1_keypoint_image'] = L.Pooling(name='pool1_keypoint_image', bottom='data_keypoint_image', pooling_param=dict(pool=P.Pooling.MAX, kernel_size=5, stride=5))
    n['flatten_keypoint_image'] = L.Flatten(name='flatten_keypoint_image', bottom='pool1_keypoint_image')

    # Keypoint class features
    n['flatten_keypoint_class'] = L.Flatten(name='flatten_keypoint_class', bottom='data_keypoint_class')

    # Fusion
    n['concat'] = L.Concat(name='concat', bottom=[fc7_out_name, 'flatten_keypoint_image', 'flatten_keypoint_class'])
    fc8_out_name = add_wrapped_fc_layer(n, 'fc8', 'concat', 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc9_out_name = add_wrapped_fc_layer(n, 'fc9', fc8_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc9_out_name)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()

def create_model_o(lmdb_paths, batch_size):
    fc7_lmdb_path = lmdb_paths[0]
    conv3_cols_lmdb_path = lmdb_paths[1]
    label_lmdb_path = lmdb_paths[2]

    n = caffe.NetSpec()
    # Data layers
    n['fc7'] = L.Data(name='fc7', batch_size=batch_size, backend=P.Data.LMDB, source=fc7_lmdb_path)
    n['fc7_flatten'] = L.Flatten(name='fc7_flatten', bottom='fc7')
    n['conv3_cols'] = L.Data(name='conv3_cols', batch_size=batch_size, backend=P.Data.LMDB, source=conv3_cols_lmdb_path)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # Conv3 features
    fc1_out_name = add_wrapped_fc_layer(n, 'fc1', 'conv3_cols', 384, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc2_out_name = add_wrapped_fc_layer(n, 'fc2', fc1_out_name, 384, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Fusion
    n['concat'] = L.Concat(name='concat', bottom=['fc7_flatten', fc2_out_name])
    fc8_out_name = add_wrapped_fc_layer(n, 'fc8', 'concat', 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc9_out_name = add_wrapped_fc_layer(n, 'fc9', fc8_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc9_out_name)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()

def create_model_p(lmdb_paths, batch_size):
    fc7_lmdb_path = lmdb_paths[0]
    conv4_cols_lmdb_path = lmdb_paths[1]
    label_lmdb_path = lmdb_paths[2]

    n = caffe.NetSpec()
    # Data layers
    n['fc7'] = L.Data(name='fc7', batch_size=batch_size, backend=P.Data.LMDB, source=fc7_lmdb_path)
    n['fc7_flatten'] = L.Flatten(name='fc7_flatten', bottom='fc7')
    n['conv4_cols'] = L.Data(name='conv4_cols', batch_size=batch_size, backend=P.Data.LMDB, source=conv4_cols_lmdb_path)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # conv4 features
    fc1_out_name = add_wrapped_fc_layer(n, 'fc1', 'conv4_cols', 384, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc2_out_name = add_wrapped_fc_layer(n, 'fc2', fc1_out_name, 384, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Fusion
    n['concat'] = L.Concat(name='concat', bottom=['fc7_flatten', fc2_out_name])
    fc8_out_name = add_wrapped_fc_layer(n, 'fc8', 'concat', 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc9_out_name = add_wrapped_fc_layer(n, 'fc9', fc8_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc9_out_name)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()

def create_model_q(lmdb_paths, batch_size):
    fc7_lmdb_path = lmdb_paths[0]
    pool5_cols_lmdb_path = lmdb_paths[1]
    label_lmdb_path = lmdb_paths[2]

    n = caffe.NetSpec()
    # Data layers
    n['fc7'] = L.Data(name='fc7', batch_size=batch_size, backend=P.Data.LMDB, source=fc7_lmdb_path)
    n['fc7_flatten'] = L.Flatten(name='fc7_flatten', bottom='fc7')
    n['pool5_cols'] = L.Data(name='pool5_cols', batch_size=batch_size, backend=P.Data.LMDB, source=pool5_cols_lmdb_path)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # pool5 features
    fc1_out_name = add_wrapped_fc_layer(n, 'fc1', 'pool5_cols', 384, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc2_out_name = add_wrapped_fc_layer(n, 'fc2', fc1_out_name, 384, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Fusion
    n['concat'] = L.Concat(name='concat', bottom=['fc7_flatten', fc2_out_name])
    fc8_out_name = add_wrapped_fc_layer(n, 'fc8', 'concat', 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc9_out_name = add_wrapped_fc_layer(n, 'fc9', fc8_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc9_out_name)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()

def create_model_r(lmdb_paths, batch_size):
    pool5_weighted_lmdb_path = lmdb_paths[0]
    label_lmdb_path = lmdb_paths[1]

    n = caffe.NetSpec()
    # Data layers
    n['pool5_weighted'] = L.Data(name='pool5_weighted', batch_size=batch_size, backend=P.Data.LMDB, source=pool5_weighted_lmdb_path)
    n['label'] = L.Data(name='label', batch_size=batch_size, backend=P.Data.LMDB, source=label_lmdb_path)
    n['label_class'], n['label_azimuth'], n['label_elevation'], n['label_tilt'] = L.Slice(name='labe-slice', bottom='label', ntop=4, slice_param=dict(
        slice_dim=1, slice_point=[1,2,3]
    ))
    n['silence-label_class'] = L.Silence(name='silence-label_class', bottom='label_class', ntop=0)

    # FC layers
    fc6_out_name = add_wrapped_fc_layer(n, 'fc6', 'pool5_weighted', 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)
    fc7_out_name = add_wrapped_fc_layer(n, 'fc7', fc6_out_name, 4096, dropout_ratio=DEFAULT_DROPOUT_RATIO)

    # Prediction and loss layers
    add_prediction_layers(n, 'pred_', fc7_out_name)
    add_loss_acc_layers(n, ['pred_', 'label_'])

    return n.to_proto()

def create_model_r2(lmdb_paths, batch_size, crop_size=gv.g_images_resize_dim, imagenet_mean_file=gv.g_image_mean_binaryproto_file):
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


def main():
    # Potential arguments
    # sub_lmdb_names = ['image_lmdb', 'viewpoint_label_lmdb']
    # model_id = 'r4cnn'
    # create_model_fn = create_model_r4cnn
    # input_data_shapes = dict(data=(1, 3, 227, 227))

    # sub_lmdb_names = ['image_lmdb', 'viewpoint_label_lmdb']
    # model_id = 'r4cnnpp_new_blah'
    # create_model_fn = create_model_r4cnnpp

    # sub_lmdb_names = ['image_lmdb', 'gaussian_keypoint_map_lmdb', 'keypoint_class_lmdb', 'viewpoint_label_lmdb']
    # model_id = 'j_new'
    # create_model_fn = create_model_j

    # sub_lmdb_names = ['fc7_lmdb', 'conv3_cols_lmdb', 'viewpoint_label_lmdb']
    # model_id = 'o'
    # create_model_fn = create_model_o

    # sub_lmdb_names = ['fc7_lmdb', 'conv4_cols_lmdb', 'viewpoint_label_lmdb']
    # model_id = 'p'
    # create_model_fn = create_model_p

    # sub_lmdb_names = ['fc7_lmdb', 'pool5_cols_lmdb', 'viewpoint_label_lmdb']
    # model_id = 'q'
    # create_model_fn = create_model_q

    # sub_lmdb_names = ['pool5_weighted_lmdb', 'viewpoint_label_lmdb']
    # model_id = 'r'
    # create_model_fn = create_model_r
    # input_data_shapes = dict(pool5_weighted=(1, 6, 6, 256))

    sub_lmdb_names = ['image_lmdb', 'pool5_weight_maps_lmdb', 'viewpoint_label_lmdb']
    model_id = 'r2'
    create_model_fn = create_model_r2
    input_data_shapes = dict(data=(1, 3, 227, 227), pool5_weight_map=(1, 1, 6, 6))

    # sub_lmdb_names = ['image_lmdb', 'pool5_weight_maps_lmdb', 'viewpoint_label_lmdb']
    # model_id = 's'
    # create_model_fn = create_model_s
    # input_data_shapes = dict(data=(1, 3, 227, 227), pool5_weight_map=(1, 1, 6, 6))


    train_batch_size = 192
    test_batch_size = 192
    # input_data_shapes = dict(data=(1, 3, 227, 227), data_keypoint_image=(1, 1, 227, 227), data_keypoint_class=(1, 34), data_pool5=(1, 256))
    # input_data_shapes = dict(fc7=(1, 4096), conv3_cols=(1, 384), conv4_cols=(1, 384), pool5_cols=(1, 256))
    syn_stepsize = 100000
    syn_max_iter = 100000
    real_stepsize = 2000
    real_max_iter = 10000
    real_test_interval = 1000
    verify_net = False

    # Set LMDB paths
    syn_lmdb_paths = [os.path.join(gv.g_corresp_syn_lmdb_folder, lmdb_name) for lmdb_name in sub_lmdb_names]
    real_train_lmdb_paths = [os.path.join(gv.g_corresp_real_train_lmdb_folder, lmdb_name) for lmdb_name in sub_lmdb_names]
    real_test_lmdb_paths = [os.path.join(gv.g_corresp_real_test_lmdb_folder, lmdb_name) for lmdb_name in sub_lmdb_names]

    # Define paths for solver and models for training on synthetic data
    model_root = os.path.join(gv.g_corresp_model_root_folder, model_id)
    model_syn_train_path = os.path.join(model_root, 'syn-train.prototxt')
    solver_syn_path = os.path.join(model_root, 'solver_syn.prototxt')

    # Define paths for solver and models for fine-tuning on real data
    model_real_train_path = os.path.join(model_root, 'real-train.prototxt')
    model_real_test_path = os.path.join(model_root, 'real-test.prototxt')
    solver_real_path = os.path.join(model_root, 'solver_real.prototxt')

    # Define path to deploy prototxt
    model_deploy_path = os.path.join(model_root, 'deploy.prototxt')

    # Make model root folder if it doesn't exist
    if not os.path.exists(model_root):
        os.makedirs(model_root)

    # Save synthetic training model prototxt
    model_syn_train = create_model_fn(syn_lmdb_paths, train_batch_size)
    with open(model_syn_train_path, 'w') as f:
        f.write(str(model_syn_train))
    # Save real training model prototxt
    model_real_train = create_model_fn(real_train_lmdb_paths, train_batch_size)
    with open(model_real_train_path, 'w') as f:
        f.write(str(model_real_train))
    # Save real test model prototxt
    model_real_test = create_model_fn(real_test_lmdb_paths, test_batch_size)
    with open(model_real_test_path, 'w') as f:
        f.write(str(model_real_test))
    # Save deploy model prototxt
    model_deploy_str = netspec_to_deploy_prototxt_str(model_syn_train, input_data_shapes)
    with open(model_deploy_path, 'w') as f:
        f.write(model_deploy_str)

    # Verify if needed
    if verify_net:
        verify_netspec(model_real_test)

    # Save synthetic data solver parameters
    syn_override_params = dict(
        train_net=model_syn_train_path,
        test_net=model_real_test_path,
        snapshot_prefix='syn',
        stepsize=syn_stepsize,
        max_iter=syn_max_iter
    )
    syn_solver_params = merge_dicts(DEFAULT_SOLVER_DICT, syn_override_params)
    syn_solver_str = dict_to_solver_text(syn_solver_params)
    with open(solver_syn_path, 'w') as f:
        f.write(syn_solver_str)

    # Save real data solver parameters
    real_override_params = dict(
        train_net=model_real_train_path,
        test_net=model_real_test_path,
        snapshot_prefix='real',
        stepsize=real_stepsize,
        max_iter=real_max_iter,
        test_interval=real_test_interval
    )
    real_solver_params = merge_dicts(DEFAULT_SOLVER_DICT, real_override_params)
    real_solver_str = dict_to_solver_text(real_solver_params)
    with open(solver_real_path, 'w') as f:
        f.write(real_solver_str)

    # Create synthetic training and real fine-tuning training scripts
    shutil.copy(os.path.join(gv.g_corresp_model_root_folder, 'train_model.sh'), os.path.join(model_root, 'syn.sh'))
    shutil.copy(os.path.join(gv.g_corresp_model_root_folder, 'train_model.sh'), os.path.join(model_root, 'real.sh'))

if __name__ == '__main__':
    main()