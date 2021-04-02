"""Learned JPEG Module learning quantization tables only.
The training code used here is based on the example code provided by the "tensorflow-compression" package.

"""



import glob
import io
from absl import app
from utils.image_utils import *
from utils.arg_parser import parse_args
from lpips_tensorflow import lpips_tf


tf.disable_v2_behavior()


class JPEGAutoEncoder(tf.keras.layers.Layer):
    """JPEG encoder and decoder with learned quantization tables and attention based editing"""

    def __init__(self, training=True, *args, **kwargs):
        self.training = training
        super(JPEGAutoEncoder, self).__init__(*args, **kwargs)
        self.dim = 8

    def build(self, input_shape):
        w_init = tf.random_uniform_initializer(minval=1e-5, maxval=2 * 1e-5)  # Initializer for quanttization tables

        self.quantizer_weights = tf.Variable(initial_value=w_init(shape=(self.dim, self.dim, 2), dtype='float32'),
                                             trainable=True,
                                             constraint=lambda t: tf.clip_by_value(t, 1e-5, 255 * 1e-5)) * 1e5 #Optimization variable for quantization tables
        self.quantizer = QuantizationLayer(self.quantizer_weights)
        self.dequantizer = QuantizationLayer(tf.reciprocal(self.quantizer_weights))
        super(JPEGAutoEncoder, self).build(input_shape)

    def call(self, tensor):
        input_image = tensor #store input image
        tensor = rgb2ycbcr(tensor * 255) #scale image to [0, 255] and convert to YCbCr
        tensor = tensor - 128 #center values around 0
        tensor = image_to_patches(tensor, 8, 8) #extract patches of size 8x8
        tensor = dct_2D(tensor) #perform 2D DCT on image
        tensor = self.quantizer(tensor)
        tensor = differentiable_round(tensor)
        tensor = self.dequantizer(tensor)
        tensor = idct_2D(tensor)
        tensor = patches_to_image(tensor)
        tensor = tensor + 128
        tensor = tf.clip_by_value(ycbcr2rgb(tensor), clip_value_min=0, clip_value_max=255) / 255
        return tensor,  self.quantizer_weights


class QuantizationLayer(tf.keras.layers.Layer):
    """Quantization Layer, performs pointwise division by a quantization table"""
    def __init__(self, weight):
        super(QuantizationLayer, self).__init__()
        self.weight = weight

    def call(self, x):
        i = tf.transpose(tf.constant([[1, 0], [0, 1], [0, 1]], dtype='float32'))
        w = tf.matmul(self.weight, i)  # duplicate chrominance table
        return tf.math.divide(x, w) #pointwise division by the quantization table


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


    # Get training patch from dataset.
    x = train_dataset.make_one_shot_iterator().get_next()

    # Instantiate model.
    model = JPEGAutoEncoder(training=True)

    # Build autoencoder.
    x_tilde,  qtables = model(x)


    # Mean squared error across pixels.
    train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
    # PSNR
    train_psnr = 10 * tf.math.log(1 / train_mse) / tf.log(tf.constant(10, dtype=train_mse.dtype))
    # Multiply by 255^2 to correct for rescaling.
    train_mse *= 255 ** 2

    # Perceptual loss
    lpips = tf.reduce_mean(lpips_tf.lpips(x, x_tilde, model='net-lin', net='alex'))

    # L1 norm over quantization tables
    summed_weights = tf.reduce_sum(tf.abs(tf.reciprocal(model.quantizer_weights)))

    # The rate-distortion loss
    train_loss = args.lmbda * (
                train_mse + args.lpips_weight * lpips) + args.total_weight_regularizer * summed_weights

    #Variable for showing the true bpp in tensorboard
    bpp = tf.Variable(0, trainable=False, dtype='float32')

    # Minimize loss and execute update op.
    step = tf.train.create_global_step()
    learning_rate = tf.placeholder(tf.float32)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(train_loss, global_step=step)

    # reshape quantization tables to format supported by PIL
    qtables = tf.transpose(model.quantizer_weights, [2, 1, 0])
    qtables = tf.round(tf.reshape(qtables, (2, 64)))



    # Tensorboard
    tf.summary.scalar("loss", train_loss)
    tf.summary.scalar("true_bpp", bpp)
    tf.summary.scalar("mse", train_mse)
    tf.summary.scalar("psnr", train_psnr)
    tf.summary.scalar("total weight", summed_weights)
    tf.summary.scalar("lpips", lpips)

    tf.summary.image("original", quantize_image(x))
    tf.summary.image("reconstruction", quantize_image(x_tilde))


    hooks = [
        tf.train.StopAtStepHook(last_step=args.last_step),
        tf.train.NanTensorHook(train_loss),
    ]
    scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=1))
    with tf.train.MonitoredTrainingSession(
            hooks=hooks, checkpoint_dir=args.checkpoint_dir,
            save_checkpoint_secs=300, save_summaries_secs=5, scaffold=scaffold) as sess:
        iter = 1 # counter for iterations
        applied_lr = args.lr # load learning rate from arguments
        while not sess.should_stop():
            outp = sess.run([train_op, qtables, x_tilde, train_mse], feed_dict={learning_rate: applied_lr})

            imgs = outp[2]
            qt = outp[1].astype('int')
            qt = np.clip(qt, 0, 255)
            if iter % 10000 == 0:
                applied_lr *= args.lr_decay

            bpp_acc = 0  # accumulator for true bpp
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
