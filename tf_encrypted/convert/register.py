import array
import logging
import numpy as np
import tensorflow as tf
import os
import yaml
from typing import Any, Dict, List
from collections import OrderedDict


from ..layers import Conv2D, Relu, Sigmoid, Dense, AveragePooling2D, MaxPooling2D
from ..protocol.pond import PondPrivateTensor, PondMaskedTensor


def registry() -> Dict[str, Any]:
    reg = {
        'Placeholder': placeholder,
        'Const': constant,
        'Conv2D': conv2d,
        'Relu': relu,
        'Sigmoid': sigmoid,
        'MatMul': matmul,
        'Shape': shape,
        'StridedSlice': strided_slice,
        'Add': add,
        'Sub': sub,
        'Transpose': transpose,
        'Reshape': reshape,
        'Pack': pack,
        'Rsqrt': rsqrt,
        'Mul': mul,
        'ExpandDims': expand_dims,
        'AvgPool': avgpool,
        'Squeeze': squeeze,
        'ConcatV2': concat,
        'BiasAdd': bias_add,
        'MaxPool': maxpool,
        'Pad': pad,
        'BatchToSpaceND': batch_to_space_nd,
        'SpaceToBatchND': space_to_batch_nd,
        'ArgMax': argmax,
        'required_space_to_batch_paddings': required_space_to_batch_paddings,
        'flatten': flatten,
        'conv2d': keras_conv2d,
        'Slice': slice,
        'Neg': negative,
        'Split': split,
        'Identity': identity,
        "GatherV2": gather,
        "dense": keras_dense,
    }

    return reg


convert_dir = os.path.dirname(os.path.abspath(__file__))
specops_path = os.path.join(convert_dir, "specops.yaml")
with open(specops_path, "r") as stream:
    loaded_yaml = yaml.load(stream, Loader=yaml.SafeLoader)
    sorted_yaml = sorted(loaded_yaml.items(), key=lambda kv: kv[0])
    REGISTERED_SPECOPS = OrderedDict(sorted_yaml)


def placeholder(converter, node: Any, inputs: List[str]) -> Any:
    return tf.placeholder(node.attr["dtype"].type,
                          shape=node.attr["shape"].shape)


def constant(converter, node: Any, inputs: List[str]) -> Any:
    # need to able to access the underlying weights return the node
    return node


def identity(converter, node: Any, inputs: List[str]) -> Any:
    # need to able to access the underlying weights return the node
    return converter.outputs[inputs[0]]


def matmul(converter, node: Any, inputs: List[str]) -> Any:
    a = converter.outputs[inputs[0]]
    b = converter.outputs[inputs[1]]

    tensor = b.attr["value"].tensor

    b_shape = [i.size for i in tensor.tensor_shape.dim]

    transpose_a = node.attr["transpose_a"].b
    transpose_b = node.attr["transpose_b"].b

    layer = Dense(a.shape.as_list(),
                  b_shape[1],
                  transpose_input=transpose_a,
                  transpose_weight=transpose_b)

    dtype = tensor.dtype

    if dtype == tf.float32:
        nums = array.array('f', tensor.tensor_content)
    elif dtype == tf.float64:
        nums = array.array('d', tensor.tensor_content)
    else:
        raise TypeError("Unsupported dtype for weights")

    inputter_fn = lambda: tf.constant(np.array(nums).reshape(b_shape))
    w = converter.protocol.define_private_input(converter.model_provider, inputter_fn)

    layer.initialize(initial_weights=w)

    return layer.forward(a)


def conv2d(converter, node, inputs):
    input = converter.outputs[inputs[0]]
    filter = converter.outputs[inputs[1]]

    if isinstance(filter, tf.NodeDef):
        shape = [i.size for i in filter.attr["value"].tensor.tensor_shape.dim]
        w = nodef_to_private_pond(converter, filter)
    else:
        shape = filter.shape.as_list()
        w = filter

    format = node.attr["data_format"].s.decode('ascii')

    layer = Conv2D(
        input.shape.as_list(), shape,
        strides=int(max(node.attr["strides"].list.i)),
        padding=node.attr["padding"].s.decode('ascii'),
        channels_first=format == "NCHW"
    )

    layer.initialize(initial_weights=w)

    out = layer.forward(input)

    return out


def keras_conv2d(converter, interiors, inputs):
    input = converter.outputs[inputs[0]]

    conv_op = interiors["Conv2D"]
    kernel = interiors["kernel"]
    k = nodef_to_private_pond(converter, kernel)
    try:
        bias = interiors["bias"]
        b = nodef_to_private_pond(converter, bias)
        for ax in [0, -1, -1]:
            b = b.expand_dims(axis=ax)
    except KeyError:
        b = None

    input_shape = input.shape.as_list()
    shape = [i.size for i in kernel.attr["value"].tensor.tensor_shape.dim]
    format = conv_op.attr["data_format"].s.decode('ascii')
    strides = int(max(conv_op.attr["strides"].list.i))
    padding = conv_op.attr["padding"].s.decode('ascii')

    layer = Conv2D(
        input_shape, shape,
        strides=strides,
        padding=padding,
        channels_first=format == "NCHW"
    )

    layer.initialize(initial_weights=k, initial_bias=b)
    out = layer.forward(input)

    return out


def keras_dense(converter, interiors, inputs):
    input = converter.outputs[inputs[0]]

    kernel = interiors["kernel"]
    k = nodef_to_private_pond(converter, kernel)
    try:
        bias = interiors["bias"]
        b = nodef_to_private_pond(converter, bias)
    except KeyError:
        b = None

    input_shape = input.shape.as_list()
    shape = [i.size for i in kernel.attr["value"].tensor.tensor_shape.dim]

    layer = Dense(input_shape,
                  out_features=shape[1])

    layer.initialize(initial_weights=k, initial_bias=b)
    out = layer.forward(input)

    return out


def relu(converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]

    return Relu(input.shape.as_list()).forward(input)


def sigmoid(converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]

    return Sigmoid(input.shape.as_list()).forward(input)


def strided_slice(converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]

    if isinstance(input, tf.NodeDef):
        input_out = nodef_to_private_pond(converter, input)
    else:
        input_out = input

    begin = converter.outputs[inputs[1]]
    end = converter.outputs[inputs[2]]
    strides = converter.outputs[inputs[3]]

    begin_mask = node.attr["begin_mask"].i
    end_mask = node.attr["end_mask"].i
    ellipsis_mask = node.attr["ellipsis_mask"].i
    new_axis_mask = node.attr["new_axis_mask"].i
    shrink_axis_mask = node.attr["shrink_axis_mask"].i

    begin = tf.constant(begin.attr["value"].tensor)
    end = tf.constant(end.attr["value"].tensor)
    strides = tf.constant(strides.attr["value"].tensor)

    return converter.protocol.strided_slice(input_out, begin, end, strides=strides,
                                            begin_mask=begin_mask,
                                            end_mask=end_mask,
                                            ellipsis_mask=ellipsis_mask,
                                            new_axis_mask=new_axis_mask,
                                            shrink_axis_mask=shrink_axis_mask)


def pack(converter, node: Any, inputs: List[str]) -> Any:
    final_inputs = []

    for input in inputs:
        input_c = converter.outputs[input]
        if isinstance(input_c, tf.NodeDef):
            final_inputs.append(nodef_to_private_pond(converter, input_c))
        else:
            final_inputs.append(input_c)

    return converter.protocol.stack(final_inputs, axis=node.attr["axis"].i)


def bias_add(converter, node: Any, inputs: List[str]) -> Any:
    a = converter.outputs[inputs[0]]
    b = converter.outputs[inputs[1]]

    if isinstance(a, tf.NodeDef):
        a_out = nodef_to_private_pond(converter, a)
    else:
        a_out = a

    if isinstance(b, tf.NodeDef):
        b_out = nodef_to_private_pond(converter, b)
    else:
        b_out = b

    return converter.protocol.add(a_out, b_out)


def maxpool(converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]

    ksize = node.attr["ksize"].list.i
    s = node.attr["strides"].list.i

    padding = node.attr["padding"].s.decode('ascii')
    pool_size = [ksize[1], ksize[2]]
    strides = [s[1], s[2]]

    shape = [int(i) for i in input.shape]

    channels_first = node.attr["data_format"].s.decode('ascii') == "NCHW"

    max = MaxPooling2D(shape, pool_size, strides, padding, channels_first)

    out = max.forward(input)

    return out


def shape(converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]

    return input.shape


def reshape(converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]
    shape = converter.outputs[inputs[1]]

    tensor = shape.attr["value"].tensor
    dtype = shape.attr["dtype"].type
    if dtype == tf.int32:
        nums = array.array('i', tensor.tensor_content)
    elif dtype == tf.int64:
        nums = array.array('l', tensor.tensor_content)
    else:
        raise TypeError("Unsupported dtype for reshape shape")

    return converter.protocol.reshape(input, list(nums))


def transpose(converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]
    perm = converter.outputs[inputs[1]]

    tensor = perm.attr["value"].tensor
    shape = [i.size for i in tensor.tensor_shape.dim]

    dtype = perm.attr["dtype"].type
    if dtype == tf.int32:
        nums = array.array('i', tensor.tensor_content)
    elif dtype == tf.int64:
        nums = array.array('l', tensor.tensor_content)
    else:
        raise TypeError("Unsupported dtype for transpose perm")

    return converter.protocol.transpose(input, np.array(nums).reshape(shape))


def expand_dims(converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]

    if isinstance(input, tf.NodeDef):
        input_out = nodef_to_private_pond(converter, input)
    else:
        input_out = input

    input_axis = converter.outputs[inputs[1]]
    axis_attr = input_axis.attr["value"].tensor.int_val
    axis_val = array.array('i', axis_attr)[0]

    return converter.protocol.expand_dims(input_out, axis_val)


def negative(converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]

    if isinstance(input, tf.NodeDef):
        input_out = nodef_to_private_pond(converter, input)
    else:
        input_out = input

    return converter.protocol.negative(input_out)


def squeeze(converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]

    axis = node.attr["squeeze_dims"].list.i

    return converter.protocol.squeeze(input, list(axis))


def gather(converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]
    indices = converter.outputs[inputs[1]]
    axis = converter.outputs[inputs[2]]

    if isinstance(input, tf.NodeDef):
        input_out = nodef_to_private_pond(converter, input)
    else:
        input_out = input

    indices_out = list(nodef_to_numpy_array(indices))

    axis_val = axis.attr["value"].tensor.int_val[0]

    return converter.protocol.gather(input_out, indices_out, axis_val)


def split(converter, node: Any, inputs: List[str]) -> Any:
    axis = converter.outputs[inputs[0]]
    input = converter.outputs[inputs[1]]

    if isinstance(input, tf.NodeDef):
        input_out = nodef_to_private_pond(converter, input)
    else:
        input_out = input

    num_split = node.attr["num_split"].i
    axis_val = axis.attr["value"].tensor.int_val[0]

    return converter.protocol.split(input_out, num_split, axis_val)[0]


def pad(converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]
    p = (converter.outputs[inputs[1]])

    paddings_t = p.attr["value"].tensor

    paddings_arr = list(array.array('I', paddings_t.tensor_content))
    paddings_lst = [paddings_arr[i:i + 2] for i in range(0, len(paddings_arr), 2)]

    return converter.protocol.pad(input, paddings_lst)


def rsqrt(converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]

    if isinstance(input, tf.NodeDef):
        tensor = input.attr["value"].tensor
        shape = [i.size for i in tensor.tensor_shape.dim]

        dtype = input.attr["dtype"].type
        if dtype == tf.float32:
            nums = array.array('f', tensor.tensor_content)
        elif dtype == tf.float64:
            nums = array.array('d', tensor.tensor_content)

        else:
            raise TypeError("Unsupported dtype for rsqrt")

        inputter_fn = lambda: tf.constant(1 / np.sqrt(np.array(nums).reshape(shape)))
    else:
        # XXX this is a little weird but the input into rsqrt is public and
        # being used only for batchnorm at the moment
        decoded = converter.protocol._decode(input.value_on_0, True)

        inputter_fn = lambda: tf.rsqrt(decoded)

    x = converter.protocol.define_public_input(converter.model_provider, inputter_fn)

    return x


def add(converter, node: Any, inputs: List[str]) -> Any:
    a = converter.outputs[inputs[0]]
    b = converter.outputs[inputs[1]]

    if isinstance(a, tf.NodeDef):
        a_out = nodef_to_public_pond(converter, a)
    else:
        a_out = a

    if isinstance(b, tf.NodeDef):
        b_out = nodef_to_public_pond(converter, b)
    else:
        b_out = b

    return converter.protocol.add(a_out, b_out)


def sub(converter, node: Any, inputs: List[str]) -> Any:
    a = converter.outputs[inputs[0]]
    b = converter.outputs[inputs[1]]

    if isinstance(a, tf.NodeDef):
        a_out = nodef_to_public_pond(converter, a)
    else:
        a_out = a

    if isinstance(b, tf.NodeDef):
        b_out = nodef_to_public_pond(converter, b)
    else:
        b_out = b

    return converter.protocol.sub(a_out, b_out)


def mul(converter, node: Any, inputs: List[str]) -> Any:
    a = converter.outputs[inputs[0]]
    b = converter.outputs[inputs[1]]

    if isinstance(a, tf.NodeDef):
        a_out = nodef_to_public_pond(converter, a)
    else:
        a_out = a

    if isinstance(b, tf.NodeDef):
        b_out = nodef_to_public_pond(converter, b)
    else:
        b_out = b

    return converter.protocol.mul(a_out, b_out)


def avgpool(converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]

    ksize = node.attr["ksize"].list.i
    s = node.attr["strides"].list.i

    padding = node.attr["padding"].s.decode('ascii')
    pool_size = [ksize[1], ksize[2]]
    strides = [s[1], s[2]]

    shape = [int(i) for i in input.shape]

    channels_first = node.attr["data_format"].s.decode('ascii') == "NCHW"

    avg = AveragePooling2D(shape, pool_size, strides, padding, channels_first)

    out = avg.forward(input)

    return out


def concat(converter, node: Any, inputs: List[str]) -> Any:
    input0 = converter.outputs[inputs[0]]
    input1 = converter.outputs[inputs[1]]
    axis = converter.outputs[inputs[2]]

    return converter.protocol.concat([input0, input1], axis.attr["value"].tensor.int_val[0])


def batch_to_space_nd(converter, node, inputs):
    input = converter.outputs[inputs[0]]
    block_shape = converter.outputs[inputs[1]].attr["value"].tensor
    crops = converter.outputs[inputs[2]].attr["value"].tensor

    return converter.protocol.batch_to_space_nd(input, block_shape, crops)


def space_to_batch_nd(converter, node, inputs):
    input = converter.outputs[inputs[0]]
    block_shape = converter.outputs[inputs[1]].attr["value"].tensor
    paddings = converter.outputs[inputs[2]].attr["value"].tensor

    return converter.protocol.space_to_batch_nd(input, block_shape, paddings)


def flatten(converter, node, inputs):
    input = converter.outputs[inputs[0]]

    shape = input.shape.as_list()
    non_batch = 1
    for dim in shape[1:]:
        non_batch *= dim

    return converter.protocol.reshape(input, [-1, non_batch])


def required_space_to_batch_paddings(converter, node, inputs: List[str]):

    inputs_node = [converter.outputs[inputs[i]] for i in range(len(inputs))]
    inputs_int32 = []
    for input in inputs_node:
        pvt_check = isinstance(input, PondPrivateTensor)
        msk_check = isinstance(input, PondMaskedTensor)
        if pvt_check or msk_check:
            msg = "Revealing private input: required_space_to_batch_paddings assumes public input."
            logging.warning(msg)
            inputs_int32.append(tf.cast(input.reveal().decode(), tf.int32))
        elif isinstance(input, tf.NodeDef):
            inputs_int32.append(nodef_to_numpy_array(input))
        else:
            msg = "Unexpected input of type {}."
            raise TypeError(msg.format(type(input)))

    if len(inputs_int32) == 2:
        input_shape, block_shape = inputs_int32

        def inputter_pad():
            pads, _ = tf.required_space_to_batch_paddings(input_shape, block_shape)
            return tf.cast(pads, tf.float64)

        def inputter_crop():
            _, crops = tf.required_space_to_batch_paddings(input_shape, block_shape)
            return tf.cast(crops, tf.float64)
    else:
        base_paddings, input_shape, block_shape = inputs_int32

        def inputter_pad():
            pads, _ = tf.required_space_to_batch_paddings(input_shape,
                                                          block_shape,
                                                          base_paddings=base_paddings)
            return tf.cast(pads, tf.float64)

        def inputter_crop():
            _, crops = tf.required_space_to_batch_paddings(input_shape,
                                                           block_shape,
                                                           base_paddings=base_paddings)
            return tf.cast(crops, tf.float64)

    pad_private = converter.protocol.define_public_input(converter.model_provider, inputter_pad)
    crop_private = converter.protocol.define_public_input(converter.model_provider, inputter_crop)

    return (pad_private, crop_private)


def argmax(converter, node, inputs):
    input = converter.outputs[inputs[0]]
    axis = converter.outputs[inputs[1]].attr["value"].tensor.int_val[0]

    return converter.protocol.argmax(input, axis=axis)


def slice(converter, node, inputs):
    input = converter.outputs[inputs[0]]
    begin = nodef_to_numpy_array(converter.outputs[inputs[1]])
    size = nodef_to_numpy_array(converter.outputs[inputs[2]])

    if isinstance(input, tf.NodeDef):
        input_out = nodef_to_private_pond(converter, input)
    else:
        input_out = input

    # Slice is a special case of strided_slice. Slice takes size (the number of
    # elements we want to slice) as an input. However strided_slice takes end
    # (integer until which the slicing takes place) as input.
    # We can infere the end parameter with : end[i] = begin[i] + size[i].
    # If size is negative, the stepping go towards smaller indices.
    # In this case we can infer the end parameter with: end[i] = input_shape[i] - size[i] + 1
    end = np.zeros(len(begin))
    input_shape = input.shape.as_list()

    # if size is negative take the input dimension
    for i in range(len(end)):
        if size[i] < 0:
            end[i] = input_shape[i] - size[i] + 1
        else:
            end[i] = begin[i] + size[i]

    return converter.protocol.strided_slice(input_out, begin, end)


def nodef_to_public_pond(converter, x):
    dtype = x.attr["dtype"].type
    x_shape = [i.size for i in x.attr["value"].tensor.tensor_shape.dim]

    if len(x_shape) == 0:
        if dtype == tf.float32:
            nums = x.attr["value"].tensor.float_val
        elif dtype == tf.float64:
            nums = x.attr["value"].tensor.float_val
        elif dtype == tf.int32:
            nums = x.attr["value"].tensor.int_val
        else:
            raise TypeError("Unsupported dtype")

        def inputter_fn():
            return tf.constant(np.array(nums).reshape(1, 1))

    else:
        if dtype == tf.float32:
            nums = array.array('f', x.attr["value"].tensor.tensor_content)
        elif dtype == tf.float64:
            nums = array.array('d', x.attr["value"].tensor.tensor_content)
        elif dtype == tf.int32:
            nums = array.array('i', x.attr["value"].tensor.tensor_content)
        else:
            raise TypeError("Unsupported dtype")

        def inputter_fn():
            return tf.constant(np.array(nums).reshape(x_shape))

    x_public = converter.protocol.define_public_input(converter.model_provider, inputter_fn)

    return x_public


def nodef_to_private_pond(converter, x):
    dtype = x.attr["dtype"].type
    warn_msg = "Unexpected dtype {} found at node {}"
    err_msg = "Unsupported dtype {} found at node {}"

    x_shape = [i.size for i in x.attr["value"].tensor.tensor_shape.dim]

    if len(x_shape) == 0:
        if dtype == tf.float32:
            nums = x.attr["value"].tensor.float_val
        elif dtype == tf.float64:
            nums = x.attr["value"].tensor.float_val
        elif dtype == tf.int32:
            logging.warn(warn_msg.format(dtype, x.name))
            nums = x.attr["value"].tensor.int_val
        else:
            raise TypeError(err_msg.format(dtype, x.name))

        def inputter_fn():
            return tf.constant(np.array(nums).reshape(1, 1))

    else:
        if dtype == tf.float32:
            nums = array.array('f', x.attr["value"].tensor.tensor_content)
        elif dtype == tf.float64:
            nums = array.array('d', x.attr["value"].tensor.tensor_content)
        elif dtype == tf.int32:
            logging.warn(warn_msg.format(dtype, x.name))
            nums = array.array('i', x.attr["value"].tensor.tensor_content)
        else:
            raise TypeError(err_msg.format(dtype, x.name))

        def inputter_fn():
            return tf.constant(np.array(nums).reshape(x_shape))

    x_private = converter.protocol.define_private_input(converter.model_provider, inputter_fn)

    return x_private


def nodef_to_numpy_array(x):
    dtype = x.attr["dtype"].type
    x_shape = [i.size for i in x.attr["value"].tensor.tensor_shape.dim]

    if dtype == tf.float32:
        nums = array.array('f', x.attr["value"].tensor.tensor_content)
    elif dtype == tf.float64:
        nums = array.array('d', x.attr["value"].tensor.tensor_content)
    elif dtype == tf.int32:
        nums = array.array('i', x.attr["value"].tensor.tensor_content)
    else:
        raise TypeError("Unsupported dtype")

    return np.array(nums).reshape(x_shape)
