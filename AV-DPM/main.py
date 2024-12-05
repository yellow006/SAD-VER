import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb

from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


def parse_args_and_config(sub_idx, label_idx):
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    
    # config for training process.
    parser.add_argument("--config", type=str, default='OCED.yml', help="Path to the config file")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--exp", type=str, default="exp", help="Path for saving running related data.")
    parser.add_argument("--doc", type=str, default='log_saving', help="A string for documentation purpose. Will be the name of the log folder.")
    parser.add_argument("--comment", type=str, default="", help="A string for experiment comment")
    parser.add_argument("--verbose", type=str, default="info", help="Verbose level: info | debug | warning | critical")
    
    # when 'sample' param is activated, launching image generation process from trained models.
    parser.add_argument("--sample",default=False, action="store_true", help="Whether to produce samples from the model")
    parser.add_argument("--fid", default=True, action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument("--n_samples", type=int, default=1024, help="Numbers of generating samples")

    # when 'resume_training' param is activated, load the last checkpoint to continue training process
    parser.add_argument("--resume_training", action="store_true", default=False, help="Whether to resume training")
    
    # config for sampling process
    parser.add_argument("-i", "--image_folder", type=str, default="images", help="The folder name of samples")
    parser.add_argument("--ni", action="store_true", help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument("--sample_type", type=str, default="av-dpm", help="sampling approach (generalized, ddpm_noisy, analytic or av-dpm)")
    parser.add_argument("--skip_type", type=str, default="uniform", help="skip according to (uniform or quadratic)")
    parser.add_argument("--timesteps", type=int, default=1000, help="number of steps for generating (not training)")
    parser.add_argument("--eta", type=float, default=0.5, help="eta used to control the variances of sigma")

    # config for current subject & label for training
    parser.add_argument("--subject_idx", type=int, help="the current subject index")
    parser.add_argument("--category_idx", type=int, help="the current category(label) index")

    args = parser.parse_args()

    args.exp += '/S%d_label%d'%(sub_idx, label_idx)

    args.subject_idx = sub_idx
    args.category_idx = label_idx
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    if not args.sample:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        handler2 = None
        logger.setLevel(level)

        os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
        args.image_folder = os.path.join(
            args.exp, "image_samples", args.image_folder
        )
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            if not (args.fid or args.interpolation):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input(
                        f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                    )
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.image_folder)
                    os.makedirs(args.image_folder)
                else:
                    print("Output image folder exists. Program halted.")
                    sys.exit(0)

    # add device
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cuda")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config, logger, handler1, handler2


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main(subject_idx, label_idx):
    args, config, logger, handler1, handler2 = parse_args_and_config(subject_idx, label_idx)
    print('args: ', args)
    print('\nconfig: ', config)
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    try:
        runner = Diffusion(args, config)
        if args.sample:
            runner.sample()
        else:
            runner.train()
    except Exception:
        logging.error(traceback.format_exc())


    logger.removeHandler(handler1)
    if handler2 is not None:
        logger.removeHandler(handler2)

    # return 0


if __name__ == "__main__":
    subject = 1
    for label in range(1,2):
        main(subject, label)
