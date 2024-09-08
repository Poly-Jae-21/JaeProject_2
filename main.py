from torch.utils.tensorboard import SummaryWriter
import os
from model.train import train
import yaml
import logging
import argparse
from model.build import argument
from torch.utils.tensorboard import SummaryWriter
def main(config):
    logging.info(f"Start meta-training, experiment name: {config['name']}")
    logging.info(f"config: {config}")

    writer = SummaryWriter(os.path.join(config["path"], "tb"))
    train(config, writer=writer)

if __name__ == "__main__":
    config = argument()
    main(config)