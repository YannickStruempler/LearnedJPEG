
import tensorflow.compat.v1 as tf
import numpy as np
def dct_2D(x):
    """Compute the 2 dimensional discrete cosine transform"""
    x = tf.transpose(x, [0, 5, 1, 2, 3, 4]) #switch to channel first format
    x = tf.signal.dct(x, norm='ortho')
    x = tf.transpose(x, [0, 1, 2, 3, 5, 4]) #tranpose last two axis
    x = tf.signal.dct(x, norm='ortho')
    x = tf.transpose(x, [0, 1, 2, 3, 5, 4])  # tranpose last two axis
    x = tf.transpose(x, [0, 2, 3, 4, 5, 1])  # switch to channel last format
    return x

def idct_2D(x):
    """Compute the 2 dimensional inverse discrete cosine transform"""
    x = tf.transpose(x,[0, 5, 1, 2, 3, 4]) #switch to channel first format
    x = tf.signal.idct(x, norm='ortho')
    x = tf.transpose(x, [0, 1, 2, 3, 5, 4]) #tranpose last two axis
    x = tf.signal.idct(x, norm='ortho')
    x = tf.transpose(x, [0, 1, 2, 3, 5, 4])  # transpose last two axis
    x = tf.transpose(x, [0, 2, 3, 4, 5, 1])  # switch to channel last format
    return x

def rgb2ycbcr(im):
    """Convert tensor from RGB to YCbCr"""
    T = tf.constant([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]], dtype='float32')
    ycbcr = tf.matmul(im, tf.transpose(T))
    offset = tf.constant([0, 128, 128], dtype='float32')
    ycbcr = ycbcr + offset
    return ycbcr

def ycbcr2rgb(im):
    """Convert tensor from YCbCr to RGB"""
    T = tf.constant([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]], dtype='float32')
    offset = tf.constant([0, 128, 128], dtype='float32')
    im = im - offset
    rgb = tf.matmul(im, tf.transpose(T))
    return rgb


def round8(x):
    """round to multiple of 8"""
    base = 8
    ret =  base * tf.cast(tf.math.round(x/base)-1, 'int32')
    return ret

def get_patches(image, num_patches=100, patch_size=16):
    """Get 'num_patches' random crops from the image"""
    patches = []
    for i in range(num_patches):
        patch = tf.random_crop(image, [patch_size, patch_size, 3])
        patches.append(patch)
    patches = tf.stack(patches)
    assert patches.get_shape().dims == [num_patches, patch_size, patch_size, 3]
    return patches

def read_png(filename):
    """Loads a PNG image file."""
    string = tf.read_file(filename)
    image = tf.image.decode_image(string, channels=3)
    image.set_shape([None, None, 3])
    image = tf.cast(image, tf.float32)
    image /= 255
    return image

@tf.custom_gradient
def roundNoGradient(x):
    """Rounding function that has gradient 1"""
    def grad(dy):
        return dy
    return tf.round(x), grad

def quantize_image(image):
    """scale image to [0,255] and convert to uint"""
    image = tf.round(image * 255)
    image = tf.saturate_cast(image, tf.uint8)
    return image


def differentiable_round(z, training=True):
    """Differentiable round based on a 3rd order polynomial approximation"""
    if training:
        z_rounded = tf.round(z)
        return roundNoGradient(z_rounded + (z - z_rounded) ** 3)
    else:
        return tf.round(z)



def write_png(filename, image):
    """Saves an image to a PNG file."""
    image = quantize_image(image)
    string = tf.image.encode_png(image)
    return tf.write_file(filename, string)


def image_to_patches(image, patch_height, patch_width):
    """Convert image tensor to patches"""
    batch_size = tf.shape(image)[0]
    image_height = tf.cast(tf.shape(image)[1], dtype=tf.float32)
    image_width = tf.cast(tf.shape(image)[2], dtype=tf.float32)

    #Round to lower multiple of 8
    height = tf.cast(tf.ceil(image_height / patch_height) * patch_height, dtype=tf.int32)
    width = tf.cast(tf.ceil(image_width / patch_width) * patch_width, dtype=tf.int32)

    num_rows = height // patch_height
    num_cols = width // patch_width

    image = tf.squeeze(tf.image.resize_image_with_crop_or_pad(image, height, width))

    # Patching
    image = tf.reshape(image, [batch_size, num_rows, patch_height, width, 3])
    image = tf.transpose(image, [0, 1, 3, 2, 4])
    image = tf.reshape(image, [batch_size, num_rows, num_cols, patch_width, patch_height, 3])
    return tf.transpose(image, [0, 1, 2, 4, 3, 5])


def patches_to_image(patches):
    """Convert image patches to image tensor"""
    batch_size = tf.shape(patches)[0]
    patch_height = tf.shape(patches)[3]
    patch_width = tf.shape(patches)[4]
    num_rows = tf.shape(patches)[1]
    num_cols = tf.shape(patches)[2]
    image_height = patch_height * num_rows
    image_width = patch_width * num_cols
    # Undo patching
    image = tf.transpose(patches, [0, 1, 2, 4, 3, 5])
    image = tf.reshape(image, [batch_size, num_rows, image_width, patch_height, 3])
    image = tf.transpose(image, [0, 1, 3, 2, 4])
    image = tf.reshape(image, [batch_size, image_height, image_width, 3])

    return image



@tf.function
def addGaussianNoise(input, minstddev=0.0, maxstddev=1.0, training=True):
    """Add Gaussian noise with randomly chosen standard deviation"""
    if training:
        stddev = tf.random_uniform(shape=[1], minval=minstddev, maxval=maxstddev)
        noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=stddev, dtype=tf.float32)
        ret = input + noise
    else:
        ret = input
    return ret

@tf.function
def get_std_jpeg_qtable(qual):
    """Get a standard IJG quantization table based on a quality factor"""
    # Base qtable according to IJG standard
    base_lum = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [
        14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62], [
             18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], [
             49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], dtype='float32')
    base_chrom = np.array([[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99], [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]], dtype='float32')

    # Rescale Quantization tables
    S = tf.cond(tf.greater(np.array([50.0], dtype='float32'), qual), lambda: 5000.0 / qual, lambda: 200.0 - 2 * qual)
    qtable_lum = tf.expand_dims(tf.round((S * base_lum + 50) / 100), axis=2)
    qtable_chrom = tf.expand_dims(tf.round((S * base_chrom + 50) / 100), axis=2)
    # Reshape to format used in the learned JPEG architecture
    qtables = tf.reshape(tf.concat([qtable_lum, qtable_chrom], axis=2), shape=(8, 8, 2))
    return qtables

