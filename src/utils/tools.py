import csv
import json
import os
import pickle
import random
import string
from shutil import copyfile, rmtree
from os.path import dirname, basename
from datetime import datetime
import torch
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def subtract_list(l1, l2):
    return list(set(l1) - set(l2))


def inf_to_zero(x):
    x[x == np.inf] = 0
    return x


def print_minimax(x):
    if isinstance(x, np.ndarray):
        return f"Min: {np.amin(x)}, Max: {np.amax(x)}"
    elif isinstance(x, torch.Tensor):
        return f"Min: {torch.min(x)}, Max: {torch.max(x)}"
    else:
        raise NotImplementedError


def print_model_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)


def normalise_matrices(m):
    def normalisation(x):
        return x / torch.max(x)

    if len(m) == 2:
        # recover matrices
        m1, m2 = m

        if (torch.sum(torch.isnan(m1)) + torch.sum(torch.isnan(m2))) > 0:
            print('distance computation returns NaNs.')
        return normalisation(m1), normalisation(m2)
    else:
        if torch.sum(torch.isnan(m)) > 0:
            print('distance computation returns NaNs.')

        return normalisation(m)


def to_tensor(things, cuda=True):
    if len(things) == 1:
        if cuda:
            return torch.tensor(things[0], requires_grad=False, dtype=torch.float).cuda()
        else:
            return torch.tensor(things[0], requires_grad=False, dtype=torch.float)
    else:
        if cuda:
            return [torch.tensor(thing, requires_grad=False, dtype=torch.float).cuda() for thing in things]
        else:
            return [torch.tensor(thing, requires_grad=False, dtype=torch.float) for thing in things]


def get_current_time():
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")

    return dt_string


def copy_file(src, dst):
    copyfile(src, dst)


def move_file(src, dst):
    copyfile(src, dst)
    os.remove(src)


def move_dir(src, dst, verbose=False):
    src_base = basename(src)
    dst_base = basename(dst)
    if src_base != dst_base:
        raise Exception("Two directories do not have a same name.")

    # MAKE SURE DESTINATION EXIST
    create_directory(dst)

    src_files = get_all_files(src)
    dst_files = [f"{dst}/{src_file}" for src_file in src_files]
    src_files = [f"{src}/{src_file}" for src_file in src_files]

    for src_file, dst_file in zip(src_files, dst_files):
        move_file(src_file, dst_file)
        if verbose:
            print(f"Moved {src_file} to {dst_file}.")

    # DELETE SOURCE
    rmtree(src)


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def write_json(l, file_name):
    with open(file_name, 'w') as file_handle:
        json.dump(l, file_handle)


def read_json(file_name):
    with open(file_name, 'r') as file_handle:
        l = json.load(file_handle)
        return l


def get_all_files(directory, keep_dir=False, sort=False):
    """
    :param keep_dir:
    :param directory: A directory
    :return: List of file names in the directory
    """

    if keep_dir:
        files = [f"{directory}/{f}" for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    else:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    if sort:
        files = sorted(files)

    return files


def get_all_paths(directory, keep_dir=False):
    if keep_dir:
        paths = [f"{directory}/{d}" for d in os.listdir(directory)]
    else:
        paths = [d for d in os.listdir(directory)]
    return paths


def get_all_dirs(directory, keep_dir=False):
    if keep_dir:
        paths = [f"{directory}/{d}" for d in os.listdir(directory) if os.path.isdir(f"{directory}/{d}")]
    else:
        paths = [d for d in os.listdir(directory) if os.path.isdir(f"{directory}/{d}")]
    return paths


def rename_files_in_directory(directory, prefix, suffix):
    """
    This function renames (by enumerating) all files in a directory
    E.g.:
    If:
    - prefix = 'karyotype'
    - suffix = '.bmp'
    then:
        '123132', '12312', '2132' --> karyotype_1.bmp, karyotype_2.bmp, karyotype_3.bmp
    :param directory:
    :param prefix:
    :param suffix:
    :return:
    """
    files = get_all_files(directory)
    for i in range(len(files)):
        file_name = directory + "/" + files[i]
        new_file_name = directory + "/" + prefix + "_" + str('{:03}'.format(i + 1)) + suffix
        os.rename(file_name, new_file_name)


def check_dir_exist(directory):
    return os.path.isdir(directory)


def check_file_exist(filename):
    return os.path.exists(filename)


def delete_file(filename):
    os.remove(filename)


def delete_files_in_dir(directory):
    files = get_all_files(directory)
    for file in files:
        delete_file(directory + "/" + file)


def delete_all_in_dir(directory):
    rmtree(directory, ignore_errors=True)


def read_lines(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def write_lines(l, file_name):
    with open(file_name, 'w') as f:
        for item in l:
            f.write("%s\n" % item)


def write_list_to_csv(csvData, file_name):
    with open(file_name, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)

    csvFile.close()


def random_string(n):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))


def path_leaf(path):
    return os.path.basename(path)

    # head, tail = ntpath.split(path)
    # return tail or ntpath.basename(head)


def save_object(obj, file_name):
    # CREATE DIRECTORY IF NOT EXIST
    curr_dir = dirname(file_name)
    create_directory(curr_dir)

    with open(file_name, "wb") as file_out:
        pickle.dump(obj, file_out, pickle.HIGHEST_PROTOCOL)


def read_object(file_name):
    with open(file_name, "rb") as file_in:
        return pickle.load(file_in)


def get_all_files_in_tree(dirName):
    """
    For the given path, get the List of all files in the directory tree
    :param dirName:
    :return:
    """
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_all_files_in_tree(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles