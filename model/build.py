import argparse
import torch
import numpy as np
import random

import yaml
from utils.logger import configure_logger

def argument():
    parser = argparse.ArgumentParser(description="Using PPO for solving multiple cities planning for EVFCS placement problems")
    parser.add_argument('-n', '--name', type=str)
    parser.add_argument('-d', '--debug', action='store_true', help="run in Debug mode")
    parser.add_argument('-c', '--config', type=str, default='configs/meta_rl_ppo.yaml')
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["name"] = args.name
    config["debug"] = args.debug
    config["path"] = f"runs/{args.name}"
    configure_logger(args.name, config["path"])
    config["device"] = torch.device(config["device_id"])
    torch.autograd.set_detect_anomaly(True)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    return config
