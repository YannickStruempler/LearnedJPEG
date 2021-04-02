"""Learnable JPEG architecture used to evaluate the standard JPEG setting with sampled quantization tables
    """

from utils.image_utils import *
#tf.disable_v2_behavior()


class JPEGAutoEncoder(tf.keras.layers.Layer):
    """JPEG encoder and decoder"""

    def __init__(self, *args, **kwargs):
        self.dim = 8
        super(JPEGAutoEncoder, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.quantizer = QuantizationLayer()
        self.dequantizer = QuantizationLayer()
        super(JPEGAutoEncoder, self).build(input_shape)

    def call(self, tensor, qtables):
        qtables = tf.clip_by_value(qtables, clip_value_min=1, clip_value_max=255)
        tensor = rgb2ycbcr(tensor * 255)
        tensor = tensor - 128
        tensor = image_to_patches(tensor, 8, 8)
        tensor = dct_2D(tensor)
        tensor = self.quantizer(tensor, qtables)
        tensor = differentiable_round(tensor)
        tensor = self.dequantizer(tensor, tf.reciprocal(qtables))
        tensor = idct_2D(tensor)
        tensor = patches_to_image(tensor)
        tensor = tensor + 128
        tensor = tf.clip_by_value(ycbcr2rgb(tensor), clip_value_min=0, clip_value_max=255) / 255
        return tensor, qtables


class QuantizationLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(QuantizationLayer, self).__init__()

    def call(self, x, weight):
        i = tf.transpose(tf.constant([[1, 0], [0, 1], [0, 1]], dtype='float32'))
        w = tf.matmul(weight, i)  # duplicate chrominance table
        return tf.div(x, w)




