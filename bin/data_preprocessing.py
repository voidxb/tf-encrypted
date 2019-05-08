"""Extracting tf.data processing pipelines from full model GraphDefs."""
import os

import tensorflow as tf
# tf.graph_util.extract_sub_graph will be removed in future tf version
try:
  from tensorflow.compat.v1.graph_util import extract_sub_graph
except ImportError:
  from tensorflow.graph_util import extract_sub_graph


def data_prep_from_saved_model(
    graph_def,
    data_filenames,
    batch_size,
    data_prep_start_node="serialized_example:0",
    data_prep_end_node="DatasetToSingleElement:0"
):
  """Main function to extract data processing pipelines."""

  # Trim graph to keep only the nodes related to data pre-processing
  data_prep_end_node_name = data_prep_end_node.split(":")[0]
  gdef_trimmed = extract_sub_graph(
      graph_def,
      dest_nodes=[data_prep_end_node_name],
  )

  # Load TFRecord files then generate a Dataset of batch
  dataset = tf.data.TFRecordDataset(data_filenames)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  dataset_b = iterator.get_next()

  # Preprocess data
  data_out, = tf.import_graph_def(
      gdef_trimmed,
      input_map={data_prep_start_node: dataset_b},
      return_elements=[data_prep_end_node],
  )

  # TFE expects tensors with fully defined shape
  fixed_shape = [batch_size] + data_out.get_shape().as_list()[1:]
  data_out = tf.reshape(data_out, fixed_shape)
  return data_out


def list_files_from_dir(directory):
  file_names_list = tf.io.gfile.listdir(directory)
  path_files_list = [os.path.join(directory, f) for f in file_names_list]
  return path_files_list
