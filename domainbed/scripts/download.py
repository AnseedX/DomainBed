# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from collections import defaultdict
from torchvision.datasets import MNIST
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import argparse
import tarfile
import shutil
import gdown
import uuid
import json
import os
import urllib

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset


# utils #######################################################################

def stage_path(data_dir, name):
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)


# VLCS ########################################################################

# Slower, but builds dataset from the original sources
#
# def download_vlcs(data_dir):
#     full_path = stage_path(data_dir, "VLCS")
#
#     tmp_path = os.path.join(full_path, "tmp/")
#     if not os.path.exists(tmp_path):
#         os.makedirs(tmp_path)
#
#     with open("domainbed/misc/vlcs_files.txt", "r") as f:
#         lines = f.readlines()
#         files = [line.strip().split() for line in lines]
#
#     download_and_extract("http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar",
#                          os.path.join(tmp_path, "voc2007_trainval.tar"))
#
#     download_and_extract("https://drive.google.com/uc?id=1I8ydxaAQunz9R_qFFdBFtw6rFTUW9goz",
#                          os.path.join(tmp_path, "caltech101.tar.gz"))
#
#     download_and_extract("http://groups.csail.mit.edu/vision/Hcontext/data/sun09_hcontext.tar",
#                          os.path.join(tmp_path, "sun09_hcontext.tar"))
#
#     tar = tarfile.open(os.path.join(tmp_path, "sun09.tar"), "r:")
#     tar.extractall(tmp_path)
#     tar.close()
#
#     for src, dst in files:
#         class_folder = os.path.join(data_dir, dst)
#
#         if not os.path.exists(class_folder):
#             os.makedirs(class_folder)
#
#         dst = os.path.join(class_folder, uuid.uuid4().hex + ".jpg")
#
#         if "labelme" in src:
#             # download labelme from the web
#             gdown.download(src, dst, quiet=False)
#         else:
#             src = os.path.join(tmp_path, src)
#             shutil.copyfile(src, dst)
#
#     shutil.rmtree(tmp_path)


def download_vlcs(data_dir):
    # Original URL: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    full_path = stage_path(data_dir, "VLCS")

    download_and_extract("https://storage.googleapis.com/kaggle-data-sets/1749462/2857535/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250811%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250811T072638Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=9870ce415a12ce8d21921f1f07c6338336550eb267b89fb33e01f68253571c77e7374571b3a446959a6cdab144e842bc2ac8bf5e2bc8324dc644e8bf7f5da3c1b7aa54652945023b6446d31c3618dd367b57e463f0732ad4e5ee0f73c706ac1563902b720b6edaa3cb572858fcc5984700135229fce360eb478c217c428a11df69586210cffcd6a1a85fe2924d1a50a08d08934498d16d6d15e7a6971f1d58f387c73bfd618c6c52ce7aab24d8a1937ad95adca2721b29fe146fdfe9241d02505518637cfe8700a6d65875ec6a7f633a487a33ac38020a469ee6327d18fc439a6ef643d429bb0c90aff375e489522a988687c5843ec6b1f1e100fe2ba40e2215",
                         os.path.join(data_dir, "VLCS.tar.gz"))


# MNIST #######################################################################

def download_mnist(data_dir):
    # Original URL: http://yann.lecun.com/exdb/mnist/
    full_path = stage_path(data_dir, "MNIST")
    MNIST(full_path, download=True)


# PACS ########################################################################

def download_pacs(data_dir):
    full_path = stage_path(data_dir, "PACS")

    # Clone the PACS dataset from GitHub
    if not os.path.isdir(full_path):
        !git clone https://github.com/MachineLearning2020/Homework3-PACS temp_pacs
        os.makedirs(data_dir, exist_ok=True)
        os.rename(os.path.join("temp_pacs", "PACS"), full_path)
        # Remove the rest of the repo after moving the dataset
        import shutil
        shutil.rmtree("temp_pacs")



# Office-Home #################################################################

def download_office_home(data_dir):
    # Original URL: http://hemanthdv.org/OfficeHome-Dataset/
    full_path = stage_path(data_dir, "office_home")

    download_and_extract("https://drive.usercontent.google.com/download?id=0B81rNlvomiwed0V1YUxQdC1uOTg&export=download&authuser=0&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw&confirm=t&uuid=8c5eff08-97b6-4305-8a81-40ea0f18e85c&at=AN8xHorLyTxcuicjZc5DIrXTuYlt%3A1754896959762",
                         os.path.join(data_dir, "office_home.zip"))

    os.rename(os.path.join(data_dir, "OfficeHomeDataset_10072016"),
              full_path)


# DomainNET ###################################################################

def download_domain_net(data_dir):
    # Original URL: http://ai.bu.edu/M3SDA/
    full_path = stage_path(data_dir, "domain_net")

    urls = [
        "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"
    ]

    for url in urls:
        download_and_extract(url, os.path.join(full_path, url.split("/")[-1]))

    with open("domainbed/misc/domain_net_duplicates.txt", "r") as f:
        for line in f.readlines():
            try:
                os.remove(os.path.join(full_path, line.strip()))
            except OSError:
                pass


# TerraIncognita ##############################################################

def download_terra_incognita(data_dir):
    # Original URL: https://beerys.github.io/CaltechCameraTraps/
    # New URL: http://lila.science/datasets/caltech-camera-traps

    full_path = stage_path(data_dir, "terra_incognita")

    download_and_extract(
        "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz",
        os.path.join(full_path, "terra_incognita_images.tar.gz"))


    download_and_extract(
        "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz",
        os.path.join(full_path, "eccv_18_annotations.tar.gz"))


    include_locations = ["38", "46", "100", "43"]

    include_categories = [
        "bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum", "rabbit",
        "raccoon", "squirrel"
    ]

    images_folder = os.path.join(full_path, "eccv_18_all_images_sm/")
    annotations_folder = os.path.join(full_path,"eccv_18_annotation_files/")
    cis_test_annotations_file = os.path.join(full_path, "eccv_18_annotation_files/cis_test_annotations.json")
    cis_val_annotations_file =   os.path.join(full_path, "eccv_18_annotation_files/cis_val_annotations.json")
    train_annotations_file =   os.path.join(full_path, "eccv_18_annotation_files/train_annotations.json")
    trans_test_annotations_file =   os.path.join(full_path, "eccv_18_annotation_files/trans_test_annotations.json")
    trans_val_annotations_file =   os.path.join(full_path, "eccv_18_annotation_files/trans_val_annotations.json")
    annotations_file_list = [cis_test_annotations_file, cis_val_annotations_file, train_annotations_file, trans_test_annotations_file, trans_val_annotations_file]
    destination_folder = full_path

    stats = {}
    data = defaultdict(list)

    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)

    for annotations_file in annotations_file_list:
        annots = {}
        with open(annotations_file, "r") as f:
            annots = json.load(f)
            for k, v in annots.items():
                data[k].extend(v)



    category_dict = {}
    for item in data['categories']:
        category_dict[item['id']] = item['name']

    for image in data['images']:
        image_location = str(image['location'])

        if image_location not in include_locations:
            continue

        loc_folder = os.path.join(destination_folder,
                                  'location_' + str(image_location) + '/')

        if not os.path.exists(loc_folder):
            os.mkdir(loc_folder)

        image_id = image['id']
        image_fname = image['file_name']

        for annotation in data['annotations']:
            if annotation['image_id'] == image_id:
                if image_location not in stats:
                    stats[image_location] = {}

                category = category_dict[annotation['category_id']]

                if category not in include_categories:
                    continue

                if category not in stats[image_location]:
                    stats[image_location][category] = 0
                else:
                    stats[image_location][category] += 1

                loc_cat_folder = os.path.join(loc_folder, category + '/')

                if not os.path.exists(loc_cat_folder):
                    os.mkdir(loc_cat_folder)

                dst_path = os.path.join(loc_cat_folder, image_fname)
                src_path = os.path.join(images_folder, image_fname)

                shutil.copyfile(src_path, dst_path)

    shutil.rmtree(images_folder)
    shutil.rmtree(annotations_folder)



# SVIRO #################################################################

def download_sviro(data_dir):
    # Original URL: https://sviro.kl.dfki.de
    full_path = stage_path(data_dir, "sviro")

    download_and_extract("https://sviro.kl.dfki.de/?wpdmdl=1731",
                         os.path.join(data_dir, "sviro_grayscale_rectangle_classification.zip"))

    os.rename(os.path.join(data_dir, "SVIRO_DOMAINBED"),
              full_path)


# SPAWRIOUS #############################################################

def download_spawrious(data_dir, remove=True):
    dst = os.path.join(data_dir, "spawrious.tar.gz")
    urllib.request.urlretrieve('https://www.dropbox.com/s/e40j553480h3f3s/spawrious224.tar.gz?dl=1', dst)
    tar = tarfile.open(dst, "r:gz")
    tar.extractall(os.path.dirname(dst))
    tar.close()
    if remove:
        os.remove(dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    # download_mnist(args.data_dir)
    # download_pacs(args.data_dir)
    # download_office_home(args.data_dir)
    # download_domain_net(args.data_dir)
    # download_vlcs(args.data_dir)
    download_terra_incognita(args.data_dir)
    # download_spawrious(args.data_dir)
    # download_sviro(args.data_dir)
    # Camelyon17Dataset(root_dir=args.data_dir, download=True)
    # FMoWDataset(root_dir=args.data_dir, download=True)
