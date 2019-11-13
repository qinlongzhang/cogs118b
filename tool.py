import os
import cv2
import math
import pydicom
import glob
import numpy as np
import pandas as pd
from skimage.transform import resize
from sklearn.metrics import roc_auc_score, mean_absolute_error

############################################################
#  Utility
############################################################

def get_image_fps(image_dir):
    return glob.glob(image_dir+'/'+'*.*')


def get_extension(image_fp):
    return os.path.splitext(os.path.basename(image_fp))[1]


def get_id(image_fp):
    return os.path.splitext(os.path.basename(image_fp))[0]


def read_image(image_fp, grayscale, target_depth=None):
    """ Read using pydicom or openCV 
    image_fp: Filepath of a GRAY image

    Returns:
    image: Image with customized depth
    """
    SUPPORTED_FORMAT = [".bmp",".jpg",".png",".dcm", ".tif"]
    assert os.path.exists(image_fp), f"File does not exist {image_fp}"
    ext = get_extension(image_fp)
    assert ext in SUPPORTED_FORMAT, f"Format not supported: {ext}"
    image = None
    if grayscale: # grayscale image. e.g. chest X-ray
        if ext==".dcm":
            ds = pydicom.read_file(image_fp)
            image = ds.pixel_array
        else:
            image = cv2.imread(image_fp, 0)
        if target_depth is not None and target_depth > 1:
            image = np.stack((image,)*target_depth, -1)
    else: # rgb image. e.g. fundus images
        image = cv2.imread(image_fp, 1)
        assert len(image.shape)==3
    return image


def gray2rgb(image):
    """ Convert gray image to RGB image
    """
    assert len(image.shape) == 2
    return np.stack((image,)*3, -1)    


def normalize(image):
    """ Normalize to 0-255, dtype: np.uint8
    """
    if np.min(image) < 0:
        image = image - np.min(image)

    if np.max(image) != 0:
        image = image / np.max(image)
        image = image * 255
        image = np.uint8(image)
    return image


def resize_image(image, target_size):
    """ Resize image using skimage
    target_size: Tuple of (width, height)
    """
    return resize(image,
                 (target_size[1],target_size[0]),
                 order=1,
                 mode="constant",
                 preserve_range=True)


#*** Training ***
def prepare_csv(image_dir):
    """ Note: for multi-class classification problem.
    Given a directory of image folders (folder names are class names), 
    create a corresponding csv file.
    """
    class_names = os.listdir(image_dir)
    num_classes = len(class_names)
    row =",".join(x for x in ["image_fp"]+class_names)+"\n"
    base_label = "0,"*num_classes
    base_label = base_label[:-1]
    save_fp = os.path.join(image_dir,"data.csv")
    with open(save_fp, "w") as f:
        f.write(row)
        for i in range(num_classes):
            class_dir = os.path.join(image_dir, class_names[i])
            image_fps = glob.glob(class_dir + "/*.*")
            class_label = list(base_label)
            class_label[2*i] = '1'
            class_label = "".join(class_label)
            for image_fp in image_fps:
                f.write(f"{image_fp},{class_label}\n")
    return save_fp


#*** evaluation ***
def compute_auroc(csv_gt, csv_pred, save_fp=None):
    df_gt   = pd.read_csv(csv_gt)
    df_pred = pd.read_csv(csv_pred)
    assert df_gt.columns.tolist()==df_pred.columns.tolist()
    assert df_gt.iloc[:,0].tolist()==df_pred.iloc[:,0].tolist()
    class_names = df_gt.columns.tolist()[1:]
    y_gt = df_gt[class_names].as_matrix()
    y_pred = df_pred[class_names].as_matrix()
    if save_fp: f = open(save_fp, "w")
    aurocs = []
    for i in range(len(class_names)):
        score = roc_auc_score(y_gt[:, i], y_pred[:,i])
        aurocs.append(score)
        print(f"{class_names[i]},{score}")
        if save_fp: f.write(f"{class_names[i]},{score}\n")
    mean_auroc = np.mean(aurocs)
    print(f"mean auroc: {mean_auroc}")
    if save_fp:
        f.write("-------------------------\n")
        f.write(f"mean auroc: {mean_auroc}")


def compute_mae(csv_gt, csv_pred, save_fp=None):
    df_gt   = pd.read_csv(csv_gt)
    df_pred = pd.read_csv(csv_pred)
    assert df_gt.columns.tolist()==df_pred.columns.tolist()
    assert df_gt.iloc[:,0].tolist()==df_pred.iloc[:,0].tolist()
    class_names = df_gt.columns.tolist()[1:]
    y_gt = df_gt[class_names].as_matrix()
    y_pred = df_pred[class_names].as_matrix()
    if save_fp: f = open(save_fp, "w")
    mae = []
    for i in range(len(class_names)):
        score = mean_absolute_error(y_gt[:, i], y_pred[:,i])
        mae.append(score)
        print(f"{class_names[i]},{score}")
        if save_fp: f.write(f"{class_names[i]},{score}\n")
    mean_mae = np.mean(mae)
    print(f"mean mae: {mean_mae}")
    if save_fp:
        f.write("-------------------------\n")
        f.write(f"mean mae: {mean_mae}")