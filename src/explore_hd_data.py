import os
import sys
import shutil
import random

def doc_cnts_paths(data_path):
    '''
    INPUT: path to data repository
    OUTPUT: (8) specific file counts and paths based on file extensions (i.e. .tif, .xml, .txt)
    '''
    tif_cnt, xml_cnt, txt_cnt, misc_cnt = 0, 0, 0, 0
    tif_paths, xml_paths, txt_paths, misc_paths = [], [], [], []
    for dirpath, dirnames, filenames in os.walk(data_path):
        for file in filenames:
            if file.endswith('.tif'):
                tif_cnt += 1
                tif_paths.append(os.path.join(dirpath, file))
            elif file.endswith('.xml'):
                xml_cnt += 1
                xml_paths.append(os.path.join(dirpath, file))
            elif file.endswith('.txt'):
                txt_cnt += 1
                txt_paths.append(os.path.join(dirpath, file))
            else:
                misc_cnt += 1
                misc_paths.append(os.path.join(dirpath, file))
    return tif_cnt, xml_cnt, txt_cnt, misc_cnt, tif_paths, xml_paths, txt_paths, misc_paths


def recursive_files(dir):
    for path, _, fnames in os.walk(dir):
        for fname in fnames:
            yield os.path.join(path, fname)


def reservoirSample(stream, k):
    samples = []
    for i, x in enumerate(stream):
        # Generate the reservoir
        if i <= k:
            samples.append(x)
        else:
            # Randomly replace elements in the reservoir
            # with a decreasing probability.
            # Choose an integer between 0 and index
            replace = random.randint(0, i-1)
            if replace < k:
                samples[replace] = x
    yield samples



if __name__ == '__main__':
    ext_hd_wells_path = '/Volumes/Seagate Expansion Drive/Tobin Data/Wells'

    # tif_cnt, xml_cnt, txt_cnt, misc_cnt, tif_paths, xml_paths, txt_paths, misc_paths = doc_cnts_paths(ext_hd_wells_path)

    '''
    RESULTS:
    Variable         Type        Data/Info
    --------------------------------------
    doc_cnts_paths   function    <function doc_cnts_paths at 0x1041417d0>
    misc_cnt         int         4
    misc_paths       list        n=4
    os               module      <module 'os' from '/Users<...>a2/lib/python2.7/os.pyc'>
    tif_cnt          int         1277891
    tif_paths        list        n=1277891
    txt_cnt          int         0
    txt_paths        list        n=0
    wells_path       str         /Volumes/Seagate Expansion Drive/Tobin Data/Wells
    xml_cnt          int         1937
    xml_paths        list        n=1937
    '''
