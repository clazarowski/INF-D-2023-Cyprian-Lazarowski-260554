import time
import numpy
import os, glob

#Read files
def get_example_filename(path='./fill-a-pix'):
    for root, dirs, files in os.walk(path, topdown=False):
        return files

def filter_filename(files):
    examples = []
    for filename in files:
            if filename.endswith('.txt'):
                examples+=[filename]
    return examples

def get_fill_a_pix(filename):
    fill_a_pix = []
    with open(filename) as f:
        lines = f.read().splitlines()
        for line in lines:
            fill_a_pix.append(list(line))
    return fill_a_pix

def get_len(fill_a_pix):
    return len(fill_a_pix[0]), len(fill_a_pix)

filenames = filter_filename(get_example_filename())
for filename in filenames:
    fill_a_pix = get_fill_a_pix('./fill-a-pix/'+filename)
    x,y=get_len(fill_a_pix)