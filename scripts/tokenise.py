# -*- coding: utf8

import timeit

import fire
import deepcut
from tqdm import tqdm
import os

BEST_PATH = "/Users/heytitle/projects/tokenisers-for-thai/data/best-features-window-1--sampling-10-concated-validation-set.txt"

def zero_conv(total=12):

    for i in range(1, total+1):
        ly = 'conv-%d' % i
        print("Trying zeroing %s" % ly)
        os.environ['DEEPCUT_ZERO_LAYER'] = ly
        main("zero-conv-%d" % i)

def main(model="original"):
    dest = "tokenised-with-%s-model.txt" % (model)
    with open(BEST_PATH, "r") as fr, \
         open(dest, "w") as fw:
        lines = fr.readlines()
        for l in tqdm(lines):
            tokens = deepcut.tokenize(l.strip())
            fw.write("%s\n" % "|".join(tokens))
    print("Result is saved to %s" % dest)

if __name__ == "__main__":
    fire.Fire(dict(manual=main, zero_conv=zero_conv))