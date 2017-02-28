from collections import Counter
from PIL import Image
import subprocess
import multiprocessing as mp
from timeit import timeit

'''
Performs OCR of .tif files using a tesseract shell command and parallelizing the process using python's multiprocessing package.
'''

def img_info(tif_paths):
    '''
    INPUT: absolute paths to .tif documents
    OUTPUT: (1) prints counts of different compressions and dpi ranges for all .tif documents. Returns 'info' list containing absolute path to image, compression format, and dpi
    '''
    info = []
    comp_cnt, dpi_cnt = Counter(), Counter()
    for path in tif_paths:
        img = Image.open(path)
        info.append((path, img.info['compression'], img.info['dpi']))
    for desc in info:
        comp_cnt[desc[1]] += 1
        dpi_cnt[desc[2]] += 1
    print 'Compression Counts: {0} \nDPI Counts: {1}'.format(comp_cnt, dpi_cnt)

    return info


def shell_tesseract(path):
    '''
    INPUT: absolute path to .tif document
    TASK: performs OCR using tesseract from the shell. Creates a text file from the OCRd document using the same name and location as the .tif document.
    OUTPUT: None
    '''
    # tesseract automatically adds a .txt extension to the OCRd document. Name of new document is 3rd argument + .txt added by tesseract
    subprocess.call(['tesseract', path, path[:-4]])


def parallelize_OCR(tif_paths):
    '''
    INPUT: paths to .tif files.
    OUTPUT: time taken to complete task
    TASK: parallelize OCR of .tif files by calling shell_tesseract and using multiprocessing Pool.
    ISSUES: not tested as a function yet. Would like to print a progress report every 15 to 30 minutes.
    '''
    # parallelize OCR processing and time it
    pool = mp.Pool(processes=4)
    task = pool.map(shell_tesseract, tif_paths)
    
    return timeit(lambda: task, number=1)
