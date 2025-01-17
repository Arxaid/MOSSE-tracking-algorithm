# This file is part of the MOSSE Tracking Algorithm project
#
# Copyright (c) 2018 Tianhong Dai
# Copyright (c) 2023 Vladislav Sosedov

from src.mosse import mosse
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--lr', type=float, default=0.125, help='the learning rate')
parse.add_argument('--sigma', type=float, default=100, help='the sigma')
parse.add_argument('--num_pretrain', type=int, default=128, help='the number of pretrain')
parse.add_argument('--rotate', action='store_true', help='if rotate image during pre-training.')
parse.add_argument('--mode', type=bool, default=1, help='ROI search mode.')

if __name__ == '__main__':
    args = parse.parse_args()
    tracker = mosse(args)
    tracker.tracking()