import argparse
import sys
from absl.flags import argparse_flags

def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.

    parser.add_argument(
        "--checkpoint_dir", default="train",
        help="Directory where to save/load model checkpoints.")

    subparsers = parser.add_subparsers(
        title="commands", dest="command",
        help="What to do: 'train' loads training data and trains (or continues "
             "to train) a new model.")

    # 'train' subcommand.
    train_cmd = subparsers.add_parser(
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Trains (or continues to train) a new model.")
    train_cmd.add_argument(
        "--train_glob", default="images/*.png",
        help="Glob pattern identifying training data.")
    train_cmd.add_argument(
        "--batchsize", type=int, default=8,
        help="Batch size for training.")
    train_cmd.add_argument(
        "--patchsize", type=int, default=256,
        help="Size of image patches for training.")
    train_cmd.add_argument(
        "--lambda", type=float, default=0.01, dest="lmbda",
        help="Lambda for rate-distortion tradeoff.")
    train_cmd.add_argument(
        "--last_step", type=int, default=1000000,
        help="Train up to this number of steps.")
    train_cmd.add_argument(
        "--preprocess_threads", type=int, default=16,
        help="Number of CPU threads to use for parallel decoding of training "
             "images.")
    train_cmd.add_argument(
        "--lr", type=float, default=0.0001,
        help="learning rate")
    train_cmd.add_argument(
        "--lr_decay", type=float, default=1,
        help="learning rate decrease factor every 10k steps")
    train_cmd.add_argument(
        "--lpips_weight", type=float, default=0,
        help="weight on lpips")
    train_cmd.add_argument(
        "--total_weight_regularizer", type=float, default=0,
        help="Weight on total inverse weight of Qtable")




    # Parse arguments.
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args