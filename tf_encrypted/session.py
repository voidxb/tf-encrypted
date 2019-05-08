"""TF Encrypted extension of tf.Session."""
import os
from typing import List, Union
from collections import defaultdict
import logging

import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python import debug as tf_debug

from .config import RemoteConfig, get_config
from .protocol.pond import PondPublicTensor
from .tensor.factory import AbstractTensor


__tfe_events__ = bool(os.getenv('TFE_EVENTS', ""))
__tfe_trace__ = bool(os.getenv('TFE_TRACE', ""))
__tfe_debug__ = bool(os.getenv('TFE_DEBUG', ""))
__tensorboard_dir__ = str(os.getenv('TFE_EVENTS_DIR', '/tmp/tensorboard'))

_run_counter = defaultdict(int)  # type: Any

logging.basicConfig()
logger = logging.getLogger('tf_encrypted')
logger.setLevel(logging.DEBUG)


class Session(tf.Session):
  """
  Wrap a Tensorflow Session.

  See the documentation of
  `tf.Session <https://www.tensorflow.org/api_docs/python/tf/Session>`_
  for more details.

  :param Optional[tf.Graph] graph: A :class:`tf.Graph`.  Used similarly.
    This is the graph to be launched.  If nothing is specified, then the
    default graph in the session will be used.
  :param Optional[~tensorflow_encrypted.config.Config] config:  A
    :class:`Local <tf_encrypted.config.LocalConfig/>` or
    :class:`Remote <tf_encrypted.config.RemoteConfig>` config to be supplied
    when executing the graph.
  """

  def __init__(self, graph=None, config=None, target=None, **tf_config_kwargs):
    if config is None:
      config = get_config()

    default_target, config_proto = config.get_tf_config(**tf_config_kwargs)
    if target is None:
      target = default_target

    if isinstance(config, RemoteConfig):
      print("Starting session on target '{}' using config {}".format(
          target, config_proto))
    super(Session, self).__init__(target, graph, config_proto)

    global __tfe_debug__

    if __tfe_debug__:
      print('Session in debug mode')
      self = tf_debug.LocalCLIDebugWrapperSession(self)  # pylint: disable=self-cls-assignment

  def _sanitize_fetches(self, fetches) -> Union[List, tf.Tensor, tf.Operation]:
    """
    Sanitize `fetches` supplied to tfe.Session.run into
    tf.Session.run-compatible `fetches`.
    """

    if isinstance(fetches, (list, tuple)):
      return [self._sanitize_fetches(fetch) for fetch in fetches]
    if isinstance(fetches, (tf.Tensor, tf.Operation)):
      return fetches
    if isinstance(fetches, PondPublicTensor):
      return fetches.decode()
    if isinstance(fetches, AbstractTensor):
      return fetches.to_native()
    raise TypeError("Don't know how to fetch {}".format(type(fetches)))

  def run(
      self,
      fetches,
      feed_dict=None,
      tag=None,
      write_trace=False,
      output_partition_graphs=False
  ):
    # pylint: disable=arguments-differ
    """
    run(fetches, feed_dict, tag, write_trace) -> Any

    See the documentation for
    `tf.Session.run <https://www.tensorflow.org/api_docs/python/tf/Session#run>`_
    for more details.

    This method functions just as the one from tensorflow.

    The value returned by run() has the same shape as the fetches argument,
    where the leaves are replaced by the corresponding values returned by
    TensorFlow.

    :param Any fetches: A single graph element, a list of graph elements, or a
      dictionary whose values are graph elements or lists of graph elements
      (described in tf.Session.run docs).
    :param str->np.ndarray feed_dict: A dictionary that maps graph elements to
      values (described in tf.Session.run docs).
    :param str tag: An optional namespace to run the session under.
    :param bool write_trace: If true, the session logs will be dumped for use
      in Tensorboard.
    """

    sanitized_fetches = self._sanitize_fetches(fetches)

    if not __tfe_events__ or tag is None:
      fetches_out = super(Session, self).run(
          sanitized_fetches,
          feed_dict=feed_dict
      )
    else:
      session_tag = "{}{}".format(tag, _run_counter[tag])
      run_tag = os.path.join(__tensorboard_dir__, session_tag)
      _run_counter[tag] += 1

      writer = tf.summary.FileWriter(run_tag, self.graph)
      run_options = tf.RunOptions(
          trace_level=tf.RunOptions.FULL_TRACE,
          output_partition_graphs=output_partition_graphs
      )

      run_metadata = tf.RunMetadata()

      fetches_out = super(Session, self).run(
          sanitized_fetches,
          feed_dict=feed_dict,
          options=run_options,
          run_metadata=run_metadata
      )

      if output_partition_graphs:
        for i, g in enumerate(run_metadata.partition_graphs):
          tf.io.write_graph(
              g,
              logdir=os.path.join(__TENSORBOARD_DIR__, session_tag),
              name='partition{}.pbtxt'.format(i),
          )

      writer.add_run_metadata(run_metadata, session_tag)
      writer.close()

      if __tfe_trace__ or write_trace:
        tracer = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = tracer.generate_chrome_trace_format()
        trace_fname = '{}/{}.ctr'.format(__tensorboard_dir__, session_tag)
        with open(trace_fname, 'w') as f:
          f.write(chrome_trace)

    return fetches_out


def set_tfe_events_flag(monitor_events: bool = False) -> None:
  """
  set_tfe_events_flag(monitor_events)

  Set flag to enable or disable monitoring of runtime statistics for each call
  to session.run().

  :param bool monitor_events: Enable or disable stats, disabled by default.
  """
  global __tfe_events__
  if monitor_events is True:
    print(("Tensorflow encrypted is monitoring statistics for each",
           "session.run() call using a tag"))

  __tfe_events__ = monitor_events


def set_tfe_debug_flag(debug: bool = False) -> None:
  """
  set_tfe_debug_flag(debug)

  Set flag to enable or disable debugging mode for TF Encrypted.

  :param bool debug: Enable or disable debugging, disabled by default.
  """
  global __tfe_debug__
  if debug is True:
    print("Tensorflow encrypted is running in DEBUG mode")

  __tfe_debug__ = debug


def set_tfe_trace_flag(trace: bool = False) -> None:
  """
  set_tfe_trace_flag(trace)

  Set flag to enable or disable tracing in TF Encrypted.

  :param bool trace: Enable or disable tracing, disabled by default.
  """
  global __tfe_trace__
  if trace is True:
    logger.info("Tensorflow encrypted is dumping computation traces")

  __tfe_trace__ = trace


def set_log_directory(path):
  """
  set_log_directory(path)

  Sets the directory to write TensorBoard event and trace files to.

  :param str path: The TensorBoard logdir.
  """
  global __tensorboard_dir__
  if path:
    logger.info("Writing event and trace files to '%s'", path)

  __tensorboard_dir__ = path
