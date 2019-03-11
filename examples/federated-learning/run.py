import sys

import tensorflow as tf
import tf_encrypted as tfe

from convert import decode

if len(sys.argv) > 1:
    # config file was specified
    config_file = sys.argv[1]
    config = tfe.RemoteConfig.load(config_file)
    tfe.set_config(config)
    tfe.set_protocol(tfe.protocol.Pond())

session_target = sys.argv[2] if len(sys.argv) > 2 else None


class ModelOwner:

    ITERATIONS = 60000 // 30

    def __init__(self, player_name):
        self.player_name = player_name

    def _build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])
        return model

    def build_training_model(self, x, y):
        """
        This method will be called once by all data owners
        to create a local gradient computation on their machine.
        """

        model = self._build_model()

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        loss = model.loss_functions[0]

        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss_value = loss(y, y_pred)

        grads = tape.gradient(loss_value, model.trainable_variables)
        return grads

    # def _build_validation_model(self, x, y):
    #     predictions, loss, _ = self._build_model(x, y)
    #     most_likely = tf.argmax(predictions, axis=1)
    #     return most_likely, loss

    def _build_data_pipeline(self):

        def normalize(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        dataset = tf.data.TFRecordDataset(["./data/train.tfrecord"])
        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)
        dataset = dataset.batch(50)
        dataset = dataset.take(1)  # keep validating on the same items
        dataset = dataset.repeat()

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def update_model(self, *grads):
        grads = [tf.cast(grad, tf.float32) for grad in grads]
        
        model = self._build_model()
        optimizer = model.optimizer
        
        with tf.name_scope('update'):
            update_op = optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return update_op

        # with tf.name_scope('validate'):
        #     x, y = self._build_data_pipeline()
        #     y_hat, loss = self._build_validation_model(x, y)

        #     with tf.control_dependencies([update_op]):
        #         return tf.print('expect', loss, y, y_hat, summarize=50)


class DataOwner:

    BATCH_SIZE = 30

    def __init__(self, player_name, _training_model):
        self.player_name = player_name
        self._training_model = _training_model

    def _build_data_pipeline(self):

        def normalize(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        dataset = tf.data.TFRecordDataset(["./data/train.tfrecord"])
        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.BATCH_SIZE)

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def compute_gradient(self):

        with tf.name_scope('data_loading'):
            x, y = self._build_data_pipeline()

        with tf.name_scope('gradient_computation'):
            with tf.variable_scope(self.player_name):
                grads = self._training_model(x, y)

        return grads


model_owner = ModelOwner('model-owner')
data_owners = [
    DataOwner('data-owner-0', model_owner.build_training_model),
    DataOwner('data-owner-1', model_owner.build_training_model),
    DataOwner('data-owner-2', model_owner.build_training_model),
]

model_grads = zip(*(
    tfe.define_private_input(data_owner.player_name, data_owner.compute_gradient)
    for data_owner in data_owners
))

with tf.name_scope('secure_aggregation'):
    aggregated_model_grads = [
        tfe.add_n(grads) / len(grads)
        for grads in model_grads
    ]

iteration_op = tfe.define_output(model_owner.player_name, aggregated_model_grads, model_owner.update_model)

with tfe.Session(target=session_target) as sess:
    sess.run(tf.global_variables_initializer(), tag='init')

    for i in range(model_owner.ITERATIONS):
        if i % 100 == 0:
            print("Iteration {}".format(i))
            sess.run(iteration_op, tag='iteration')
        else:
            sess.run(iteration_op)
