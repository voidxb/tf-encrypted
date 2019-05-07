import tensorflow as tf


def wrap_in_variables(*tensors):
    variables = [
        tensor.factory.variable(
            tf.zeros(shape=tensor.shape, dtype=tensor.factory.native_type)
        )
        for tensor in tensors
    ]
    group_updater = tf.group(
        *[var.assign_from_same(tensor) for var, tensor in zip(variables, tensors)]
    )
    return group_updater, variables


def reachable_nodes(*nodes):
    reachable = set(nodes)
    queue = list(nodes)

    while queue:
        node = queue.pop(0)

        if isinstance(node, tf.Tensor):
            subnode = node.op
            if subnode not in reachable:
                reachable.add(subnode)
                queue.append(subnode)
            continue

        if isinstance(node, tf.Operation):
            for subnode in list(node.inputs) + list(node.control_inputs):
                if subnode not in reachable:
                    reachable.add(subnode)
                    queue.append(subnode)
            continue

        raise TypeError(
            "Don't know how to process {} of type {}".format(node, type(node)))

    return reachable