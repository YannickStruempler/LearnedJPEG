"""Learned JPEG Module with pre editing CNN as suggested by Talebi et al.,2020.
The training code used here is based on the example code provided by the "tensorflow-compression" package.

"""


import glob
import io
from absl import app
from utils.arg_parser import parse_args
from utils.image_utils import *
from tensorflow_compression import EntropyBottleneck
from utils.conv_blocks import ResConvLayer

tf.disable_v2_behavior()




def sample_qtables(minqual=8, maxqual=25, training=True, default_quality=10):
    if training:
      qual = tf.random_uniform(shape=[1], minval=minqual, maxval=maxqual)
    else:
        qual = default_quality #when evaluating choose specified default_quality instead of sampling
    return get_std_jpeg_qtable(qual), qual

class JPEGAutoEncoder(tf.keras.layers.Layer):
    """JPEG encoder and decoder with CNN pre-editing"""

    def __init__(self, training=True, *args, **kwargs):
        self.dim = 8
        self.training = training
        self.conv = ResConvLayer([64, 64, 64, 64, 128, 128, 3]) # Pre-Editing CNN
        self.quantizer = QuantizationLayer()
        self.dequantizer = QuantizationLayer()
        self.entropy_bottleneck = EntropyBottleneck(init_scale=1)
        super(JPEGAutoEncoder, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        super(JPEGAutoEncoder, self).build(input_shape)

    def call(self, tensor, default_quality):
        tensor = addGaussianNoise(tensor, minstddev=0, maxstddev=0.15, training=self.training) # add gaussian noise to the input
        qtables, qual = sample_qtables(default_quality=default_quality, training=self.training) # sample standard IJG quantization tables
        qtables = tf.clip_by_value(qtables, clip_value_min=1, clip_value_max=255)
        tensor = self.conv(tensor)
        tensor = rgb2ycbcr(tensor * 255)
        tensor = tensor - 128
        tensor = image_to_patches(tensor, 8, 8)
        tensor = dct_2D(tensor)
        tensor = self.quantizer(tensor, qtables)
        t, likelihoods = self.entropy_bottleneck(tensor, training=True)
        tensor = differentiable_round(tensor)
        tensor = self.dequantizer(tensor, tf.reciprocal(qtables))
        tensor = idct_2D(tensor)
        tensor = patches_to_image(tensor)
        tensor = tensor + 128
        tensor = tf.clip_by_value(ycbcr2rgb(tensor), clip_value_min=0, clip_value_max=255) / 255
        return tensor, likelihoods, qtables


class QuantizationLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(QuantizationLayer, self).__init__()

    def call(self, x, weight):
        i = tf.transpose(tf.constant([[1, 0], [0, 1], [0, 1]], dtype='float32'))
        w = tf.matmul(weight, i) # duplicate chrominance table
        return tf.math.divide(x, w)


def train(args):
    """Trains the model."""

    #Get dataset
    with tf.device("/cpu:0"):
        num_patches = 50  #patches per image
        buffer_size = 500  #shuffle patches from 'buffer_size' different images

        train_files = glob.glob(args.train_glob) #get train image glob

        if not train_files:
            raise RuntimeError(
                "No training images found with glob '{}'.".format(args.train_glob))

        get_patches_fn = lambda image: get_patches(image, num_patches=num_patches, patch_size=args.patchsize)

        train_dataset = (tf.data.Dataset.from_tensor_slices(train_files)
                         .repeat() #repeat dataset infinitely
                         .shuffle(buffer_size=len(train_files)) #shuffle all images
                         .map(read_png, num_parallel_calls=args.preprocess_threads)  #read images
                         .map(get_patches_fn, num_parallel_calls=args.preprocess_threads)  #extract training patches by random cropping
                         .apply(tf.data.experimental.unbatch())  #unbatch the random crops
                         .shuffle(buffer_size=buffer_size)  #shuffle the random crops
                         .batch(args.batchsize)  #rebatch the crops
                         .prefetch(1)
                         )


    num_pixels = args.batchsize * args.patchsize ** 2

    # Get training patch from dataset.
    x = train_dataset.make_one_shot_iterator().get_next()

    # Instantiate model.
    model = JPEGAutoEncoder(training=True)

    # Build autoencoder.
    x_tilde, likelihoods, qtables = model(x, default_quality = 10)

    # Total number of bits divided by number of pixels.
    train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

    # Mean squared error across pixels.
    train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
    # PSNR
    train_psnr = 10 * tf.math.log(1 / train_mse) / tf.log(tf.constant(10, dtype=train_mse.dtype))
    # Multiply by 255^2 to correct for rescaling.
    train_mse *= 255 ** 2


    # The rate-distortion cost.
    train_loss = args.lmbda * train_mse + train_bpp

    #Variable for showing the true bpp in tensorboard
    bpp = tf.Variable(0, trainable=False, dtype='float32')

    # Minimize loss and auxiliary loss, and execute update op.
    step = tf.train.create_global_step()
    learning_rate = tf.placeholder(tf.float32)
    main_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    main_step = main_optimizer.minimize(train_loss, global_step=step)

    aux_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate * 10)
    aux_step = aux_optimizer.minimize(model.entropy_bottleneck.losses[0])

    train_op = tf.group(main_step, aux_step,model.entropy_bottleneck.updates[0])

    # reshape quantization tables to format supported by PIL
    qtables = tf.transpose(qtables, [2, 1, 0])
    qtables = tf.round(tf.reshape(qtables, (2, 64)))


    tf.summary.scalar("loss", train_loss)
    tf.summary.scalar("bpp", train_bpp)
    tf.summary.scalar("true_bpp", bpp)

    tf.summary.scalar("mse", train_mse)
    tf.summary.scalar("psnr", train_psnr)
    tf.summary.image("original", quantize_image(x))
    tf.summary.image("reconstruction", quantize_image(x_tilde))

    hooks = [
        tf.train.StopAtStepHook(last_step=args.last_step),
        tf.train.NanTensorHook(train_loss),
    ]
    with tf.train.MonitoredTrainingSession(
            hooks=hooks, checkpoint_dir=args.checkpoint_dir,
            save_checkpoint_secs=300, save_summaries_secs=5) as sess:
        iter = 1  # counter for iterations
        applied_lr = args.lr  # load learning rate from arguments
        while not sess.should_stop():
            outp = sess.run([train_op, qtables, x_tilde, train_mse], feed_dict={learning_rate: applied_lr})
            bpp_acc = 0 # accumulator for true bpp
            imgs = outp[2]
            qt = outp[1].astype('int')
            qt = np.clip(qt, 0,255)
            if iter % 10000 == 0:
                applied_lr *= args.lr_decay
            # compute true bpp every 100 iterations
            if iter % 100 == 0:
                k = len(imgs)
                for i in range(k):
                    out = io.BytesIO()
                    img = tf.keras.preprocessing.image.array_to_img(
                        imgs[i], data_format='channels_last', scale=True)
                    img.save(out, format='jpeg', qtables=qt.tolist(), subsampling=0)

                    bpp_acc += out.tell() * 8 / (img.size[0] * img.size[1] * k)
                print(bpp_acc, outp[3])
                if not sess.should_stop():
                    bpp.load(bpp_acc, session=sess)
            iter = iter + 1



def main(args):
    # Invoke subcommand.
    if args.command == "train":
        train(args)



if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
