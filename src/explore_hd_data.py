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






if __name__ == '__main__':
    wells_path = '/Volumes/Seagate Expansion Drive/Tobin Data/Wells'

    tif_cnt, xml_cnt, txt_cnt, misc_cnt, tif_paths, xml_paths, txt_paths, misc_paths = doc_cnts_paths(wells_path)

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


    #folder which contains the sub directories
    source_dir = wells_path

    #list sub directories
    for root, dirs, files in os.walk(source_dir):

    #iterate through them
    for i in dirs:

        #create a new folder with the name of the iterated sub dir
        path = '/home/mrman/dataset-python/sub-train/' + "%s/" % i
        os.makedirs(path)

        #take random sample, here 3 files per sub dir
        filenames = random.sample(os.listdir('/home/mrman/dataset-python/train/' + "%s/" % i ), 3)

        #copy the files to the new destination
        for j in filenames:
            shutil.copyfile.copy2('/home/mrman/dataset-python/train/' + "%s/" % i  + j, path)
