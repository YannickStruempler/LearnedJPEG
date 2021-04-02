import tensorflow.compat.v1 as tf


class ResConvLayer(tf.keras.layers.Layer):
    """Residual Convolutional Preprocessing as used by Talebi et al., 2020"""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(ResConvLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.initial_conv = [
            tf.keras.layers.Conv2D(self.num_filters[0], (7, 7), input_shape=input_shape, padding='same'),
            tf.keras.layers.LeakyReLU(alpha=0.2)]
        self.first_block = [
            ResUnit(self.num_filters[1]),
            ResUnit(self.num_filters[2]),
            tf.keras.layers.Conv2D(self.num_filters[3], (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization()
        ]
        self.second_block = [
            tf.keras.layers.Conv2D(self.num_filters[4], (3, 3), padding='same'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Conv2D(self.num_filters[5], (3, 3), padding='same'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Conv2D(self.num_filters[6], (7, 7), padding='same')

        ]
        super(ResConvLayer, self).build(input_shape)

    def call(self, tensor):
        for layer in self.initial_conv:
            tensor = layer(tensor)
        in_tensor = tensor  # store tensor
        for layer in self.first_block:
            tensor = layer(tensor)
        tensor = in_tensor + tensor  # residual connection
        for layer in self.second_block:
            tensor = layer(tensor)
        return tensor


class ResUnit(tf.keras.layers.Layer):
    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(ResUnit, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tf.keras.layers.Conv2D(self.num_filters, (3, 3), input_shape=input_shape, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Conv2D(self.num_filters, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization()
        ]
        super(ResUnit, self).build(input_shape)

    def call(self, tensor):
        in_tensor = tensor
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor + in_tensor

class VGGAttention(tf.keras.layers.Layer):
  """VGG Attention layer"""

  def __init__(self, *args, **kwargs):
    super(VGGAttention, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    base_model = tf.keras.applications.VGG19(weights='imagenet') # use VGG19 network with weights pretrained on ImageNet
    self.reduce_dim = tf.keras.layers.Conv2D(128, kernel_size=(1,1), padding='same')
    self.features_extraction = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output) # extract VGG features after the 3rd max pool block
    super(VGGAttention, self).build(input_shape)

  def call(self, dct_coeff, image):
    batch = tf.shape(dct_coeff)[0]
    width = tf.shape(dct_coeff)[1]
    height = tf.shape(dct_coeff)[2]
    features = self.features_extraction(image)
    features_reduced = self.reduce_dim(features)
    attention = tf.reshape(tf.sigmoid(features_reduced), shape=[batch, width, height, 8, 8, 2])
    i = tf.transpose(tf.constant([[1, 0], [0, 1], [0, 1]], dtype='float32'))
    attention = tf.matmul(attention, i)  # duplicate chrominance attention
    return dct_coeff * attention, attention # apply attention map as pointwise multiplication

