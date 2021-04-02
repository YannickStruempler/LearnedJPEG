import io
import matplotlib.pyplot as plt
import glob
from utils.image_utils import *
from models import google_baseline, ljpeg, ljpeg_attention, ljpeg_PIL
import tensorflow.compat.v1 as tf
from lpips_tensorflow import lpips_tf
from absl import app
from absl.flags import argparse_flags
import argparse
import os
import pickle
def mse_func(a,b):
    return np.mean((np.array(a, dtype='int32') - np.array(b, dtype='int32'))**2)

class Evaluator():
    def __init__(self, image_dir, checkpoint_dir, jpeg_type, experiment_name, output_path,  quality=10, save=True):
        self.image_dir = image_dir
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.jpeg_type = jpeg_type
        self.output_path = output_path
        self.quality = quality # optional quality parameter for PIL or google Baseline
        self.save = save # if true the output images will be saved
    def evaluate(self):
        """Evaluates the metrics of a trained model"""
        path = self.output_path + "/" + self.experiment_name + ".pickle"

        # if available, read metrics from existing pickle file
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                x = pickle.load(f)
                try:
                    bpp_array, mse_array, lpips_array, psnr_array, mssim_array = x
                except:
                    print("Could not load pickle file")
        else:
            data, files_list = self.get_dataset()
            # Get training patch from dataset.
            x = data.make_one_shot_iterator().get_next()

            # Instantiate model.
            if self.jpeg_type == 'ljpeg':
                model = ljpeg.JPEGAutoEncoder(training=False)
                x_tilde, qtables = model(x)
            elif self.jpeg_type == 'ljpeg_attention':
                model = ljpeg_attention.JPEGAutoEncoder(training=False)
                x_tilde, _, qtables = model(x)
            elif self.jpeg_type == 'google':
                model = google_baseline.JPEGAutoEncoder(training=False)
                x_tilde, _, qtables = model(x, self.quality)
            else:
                qtables = get_std_jpeg_qtable(self.quality)
                model = ljpeg_PIL.JPEGAutoEncoder()
                x_tilde, qtables = model(x, qtables)

            #Convert qtables to format used by PIL
            qtables = tf.transpose(qtables, [2, 1, 0])
            qtables = tf.round(tf.reshape(qtables, (2, 64)))

            # Perceptual lpips score
            lpips = lpips_tf.lpips(x, x_tilde, model='net-lin', net='alex')

            x *= 255
            x_tilde_255 = tf.round(x_tilde * 255)

            test_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde_255))
            test_psnr = tf.squeeze(tf.image.psnr(x_tilde_255, x, 255))
            test_mssim = tf.squeeze(tf.image.ssim_multiscale(x_tilde_255, x, 255))

            with tf.Session() as sess:
                # Load the latest model checkpoint and restore all parameters
                if (self.jpeg_type == 'ljpeg' or self.jpeg_type == 'google' or self.jpeg_type == 'ljpeg_attention' or self.jpeg_type == 'ljpeg_attention_qtable'):
                    latest = tf.train.latest_checkpoint(checkpoint_dir=self.checkpoint_dir)
                    print(self.checkpoint_dir)
                    tf.train.Saver().restore(sess, save_path=latest)
                bpp_array = []
                mse_array = []
                lpips_array = []
                psnr_array = []
                mssim_array = []

                for i in range(len(files_list)):
                    # evaluate the model
                    output = sess.run([x_tilde, test_mse, qtables, lpips, test_psnr, test_mssim])
                    x_tilde_out = output[0]
                    test_mse_out = output[1]
                    qtables_out = output[2].astype('int') #np.clip(output[2].astype('int'), 0, 255)
                    lpips_out = output[3]
                    test_psnr_out = output[4]
                    test_mssim_out = output[5]
                    img_vec = np.array(x_tilde_out[0]) #get image from batch

                    # get true bpp
                    bpp = self.get_bpp(img_vec, qtables_out.tolist()) #get true bpp

                    # save image to hard drive when save==True
                    if self.save == True:
                        self.save_img(img_vec, qtables_out.tolist(), files_list[i])
                    bpp_array.append(bpp)
                    mse_array.append(test_mse_out)
                    lpips_array.append(lpips_out)
                    psnr_array.append(test_psnr_out)
                    mssim_array.append(test_mssim_out)
                sess.close()
            tf.keras.backend.clear_session()
            with open(path, 'wb') as f:
                x = bpp_array, mse_array, lpips_array, psnr_array, mssim_array
                pickle.dump(x, f) # save metrics in a pickle file

        return bpp_array, mse_array, lpips_array, psnr_array, mssim_array


    def get_bpp(self, image_array, qtables):
        out = io.BytesIO() # byte buffer
        img = tf.keras.preprocessing.image.array_to_img(image_array * 255, data_format='channels_last', scale=False)
        img.save(out, format='jpeg', qtables=qtables, subsampling=0) # save image to byte buffer with specified qtables
        return out.tell() * 8 / (img.size[0] * img.size[1]) # compute bpp (out.tell() returns number of bytes)
    def save_img(self, image_array, qtables, original_file_name):
        img = tf.keras.preprocessing.image.array_to_img(image_array * 255, data_format='channels_last', scale=False)
        file_name = self.output_path + "/" + original_file_name.split('/')[-1].split('.')[0] + "_" + self.experiment_name + ".jpg"
        img.save(file_name, format='jpeg', qtables=qtables, subsampling=0)

    def get_dataset(self):

        with tf.device("/cpu:0"):
            image_glob = glob.glob(self.image_dir)

            if not image_glob:
                raise RuntimeError(
                    "No training images found with glob '{}'.".format(self.image_dir))

            # Create dataset with full sized images and batch size 1 for evaluation
            test_dataset = (tf.data.Dataset.from_tensor_slices(image_glob)
                            .map(read_png, num_parallel_calls=16)
                            .map(lambda x: tf.image.crop_to_bounding_box(x, 0, 0, round8(tf.shape(x)[0]), round8(tf.shape(x)[1])))
                            .batch(1)
                            .prefetch(1)
                            )
        return test_dataset, image_glob


def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument(
        "--batch_dir", default=".",
        help="Directory where batch of experiments are")
    parser.add_argument(
        "--jpeg_type", default="ljpeg",
        help="Jpeg architecture")
    parser.add_argument("--test_image_glob", default = ".",
        help = "Glob patttern for where the test images are")
    parser.add_argument("--google_dir", default=None,
                        help="Directory where google baseline data is")

    # Parse arguments.
    args = parser.parse_args(argv[1:])
    return args


def main(args):

    #JPEG Standard
    cmd = "mkdir {}".format(args.batch_dir + "/_output_PIL")
    os.system(cmd)
    bpp_avg_list = []
    mse_avg_list = []
    lpips_avg_list = []
    psnr_avg_list = []
    mssim_avg_list = []
    quality_list = list(range(1,10)) + list(range(10,95,5))
    for q in quality_list:
        evaluator = Evaluator(args.test_image_glob, "", 'PIL',"Q" + str(q), args.batch_dir + "/_output_PIL", quality=q, save=True)
        bpp_array, mse_array, lpips_array, psnr_array, mssim_array = evaluator.evaluate()

        # Compute average over test image for quality factor q
        bpp_avg = np.mean(bpp_array)
        mse_avg = np.mean(mse_array)
        lpips_avg = np.mean(lpips_array)
        psnr_avg = np.mean(psnr_array)
        mssim_avg = np.mean(mssim_array)

        bpp_avg_list.append(bpp_avg)
        mse_avg_list.append(mse_avg)
        lpips_avg_list.append(lpips_avg)
        psnr_avg_list.append(psnr_avg)
        mssim_avg_list.append(mssim_avg)

    plt.figure(1)
    plt.plot(bpp_avg_list, mse_avg_list,  label="JPEG")
    plt.figure(2)
    plt.plot(bpp_avg_list, lpips_avg_list,  label="JPEG")
    plt.figure(3)
    plt.plot(bpp_avg_list, psnr_avg_list, label="JPEG")
    plt.figure(4)
    # Convert MS-SSIM to log scale for plotting
    plt.plot(bpp_avg_list, [-10 * np.log10(1 - mssim) for mssim in mssim_avg_list], label="JPEG")

    #google baseline
    if args.google_dir is not None: # check if a path to a google baseline eexperiment was given in the arguments
        bpp_avg_list = []
        mse_avg_list = []
        lpips_avg_list = []
        psnr_avg_list = []
        mssim_avg_list = []
        for quality in range(5,30,5):

            name = 'google_Q' + str(quality)
            evaluator = Evaluator(args.test_image_glob, args.google_dir, 'google', name, args.batch_dir + "/_output", quality=quality, save=True)

            # Evaluate model
            bpp_array, mse_array, lpips_array, psnr_array, mssim_array = evaluator.evaluate()

            # Compute average over test image for quality factor q
            bpp_avg = np.mean(bpp_array)
            mse_avg = np.mean(mse_array)
            lpips_avg = np.mean(lpips_array)
            psnr_avg = np.mean(psnr_array)
            mssim_avg = np.mean(mssim_array)

            bpp_avg_list.append(bpp_avg)
            mse_avg_list.append(mse_avg)
            lpips_avg_list.append(lpips_avg)
            psnr_avg_list.append(psnr_avg)
            mssim_avg_list.append(mssim_avg)
        plt.figure(1)
        plt.plot(bpp_avg_list, mse_avg_list, "+", label='google')
        plt.figure(2)
        plt.plot(bpp_avg_list, lpips_avg_list, "+", label='google')
        plt.figure(3)
        plt.plot(bpp_avg_list, psnr_avg_list,"+", label='google')
        plt.figure(4)
        # Convert MS-SSIM to log scale for plotting
        plt.plot(bpp_avg_list, [-10 * np.log10(1 - mssim) for mssim in mssim_avg_list], "+", label='google')


    #Experiment
    experiments = glob.glob(args.batch_dir + "/[!_]*") # ignore files starting with an underscore  "_"
    cmd = "mkdir {}".format(args.batch_dir + "/_output")
    os.system(cmd)
    print(experiments)
    for experiment in experiments:
        runs = glob.glob(experiment + "/*/train")
        bpp_avg_list = []
        mse_avg_list = []
        lpips_avg_list = []
        psnr_avg_list = []
        mssim_avg_list = []
        for run in runs:

            ex = experiment.split('/')[-1]
            ru = run.split('/')[-2]
            name = ex + "-" + ru
            print(name)
            evaluator = Evaluator(args.test_image_glob, run, args.jpeg_type, name, args.batch_dir + "/_output")
            #Evaluate Model
            bpp_array, mse_array, lpips_array, psnr_array, mssim_array = evaluator.evaluate()

            #Average for all test images over one run (typically one setting of lambda)
            bpp_avg = np.mean(bpp_array)
            mse_avg = np.mean(mse_array)
            lpips_avg = np.mean(lpips_array)
            psnr_avg = np.mean(psnr_array)
            mssim_avg = np.mean(mssim_array)

            bpp_avg_list.append(bpp_avg)
            mse_avg_list.append(mse_avg)
            lpips_avg_list.append(lpips_avg)
            psnr_avg_list.append(psnr_avg)
            mssim_avg_list.append(mssim_avg)
        if runs:
            plt.figure(1)
            plt.plot(bpp_avg_list, mse_avg_list, "+", label=ex)
            plt.xlabel("BPP")
            plt.ylabel("MSE")
            plt.legend()

            plt.figure(2)
            plt.plot(bpp_avg_list, lpips_avg_list, "+", label=ex)
            plt.xlabel("BPP")
            plt.ylabel("LPIPS")
            plt.legend()

            plt.figure(3)
            plt.plot(bpp_avg_list, psnr_avg_list, "+", label=ex)
            plt.xlabel("BPP")
            plt.ylabel("PSNR [dB]")
            plt.legend()
            plt.figure(4)
            # Convert MS-SSIM to log scale for plotting
            plt.plot(bpp_avg_list, [-10 * np.log10(1 - mssim) for mssim in mssim_avg_list], "+", label=ex)
            plt.xlabel("BPP")
            plt.ylabel("MSSIM [dB]")
            plt.legend()



    plt.figure(1)
    plt.xlim(0,1.2)
    plt.ylim(40,400)
    plt.savefig(args.batch_dir + "/mse.jpg")
    plt.figure(2)
    plt.xlim(0,2)
    plt.ylim(0,0.6)
    plt.savefig(args.batch_dir + "/lpips.jpg")
    plt.figure(3)
    plt.savefig(args.batch_dir + "/psnr.jpg")
    plt.figure(4)
    plt.savefig(args.batch_dir + "/mssim.jpg")
    plt.show()




if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)

