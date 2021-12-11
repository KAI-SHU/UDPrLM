# -*- coding: utf-8 -*-

import torch
from supar.utils import Config
from supar.utils.logging import init_logger, logger
from supar.utils.parallel import init_device


def parse(parser):
    parser.add_argument('--path', '-p', help='path to model file')
    parser.add_argument('--conf', '-c', default='', help='path to config file')
    parser.add_argument('--device', '-d', default='-1', help='ID of GPU to use')
    parser.add_argument('--seed', '-s', default=1, type=int, help='seed for generating random numbers')
    parser.add_argument('--threads', '-t', default=16, type=int, help='max num of threads')
    parser.add_argument("--local_rank", type=int, default=-1, help='node rank for distributed training')
    parser.add_argument("--lr", type=float, default=2e-3, help='lr')
    parser.add_argument("--mu", type=float, default=0.9, help='mu')
    parser.add_argument("--nu", type=float, default=0.9, help='nu')
    parser.add_argument("--eps", type=float, default=1e-12, help='eps')
    parser.add_argument("--decay", type=float, default=0.75, help='decay')
    parser.add_argument("--decay_steps", type=int, default=5000, help='decay_steps')
    parser.add_argument("--weight_decay", type=float, default=0.0, help='weight_decay')
    parser.add_argument("--batch_size", type=int, default=1000, help='batch_size')
    parser.add_argument('--freeze', action='store_true', help='whether to freeze bert')
    
    args, unknown = parser.parse_known_args()
    args, unknown = parser.parse_known_args(unknown, args)
    args = Config.load(**vars(args), unknown=unknown)
    Parser = args.pop('Parser')

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    init_device(args.device, args.local_rank)
    init_logger(logger, f"{args.path}.{args.mode}.log")
    logger.info('\n' + str(args))

    if args.mode == 'train':
        parser = Parser.build(**args)
        parser.train(**args)
    elif args.mode == 'evaluate':
        parser = Parser.load(args.path)
        parser.evaluate(**args)
    elif args.mode == 'predict':
        parser = Parser.load(args.path)
        parser.predict(**args)
