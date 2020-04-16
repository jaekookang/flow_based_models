'''
Utilities

2020-04-16
'''

import os
import re
import shutil
from time import strftime, gmtime


def safe_mkdir(folder):
    '''Make directory if not exists'''
    if not os.path.exists(folder):
        os.mkdir(folder)


def safe_rmdir(folder):
    '''Remove directory if exists'''
    if os.path.exists(folder):
        shutil.rmtree(folder)


def get_datetime():
    '''Get current date & time both'''
    return strftime('%Y-%m-%d_%H_%M_%S', gmtime())


def find_elements(pattern, list_data):
    '''Find elements in a list
    Args:
    - pattern: string to match in `list_data`
    - list_data: python list including elements
    Returns:
    - index: list of index; empty list if no patterns found
    - elements: list of matched elements; empty list if no patterns found
    '''
    elements = []
    index = []

    for i, l in enumerate(list_data):
        if re.search(pattern, l):
            elements.append(list_data[i])
            index.append(i)
    return index, elements
