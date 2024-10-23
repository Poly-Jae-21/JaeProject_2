import os
from model.train_v2 import train_meta_worker
from model.build import argument
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.logger import configure_logger
import torch.multiprocessing as mp

def main(args):
    args = args
    path = f"runs/{args.name}"
    configure_logger(args.name, path)
    writer = SummaryWriter(os.path.join(args["path"], "tb"))

    use_cuda = args.cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    mp.set_start_method('spawn', True)

    processes = []
    for rank in range(args.n_workers):
        p = mp.Process(target=train_meta_worker, args=(rank, args, writer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    config = argument()
    main(config)