import cv2
import os
import numpy as np
from tqdm import trange
import json

def cut_rotated_box(image, box):
    center, size, angle = box
    center, size = tuple(map(int, center)), tuple(map(int, size))

    height, width = image.shape[0], image.shape[1]

    M = cv2.getRotationMatrix2D(center, angle, 1)
    image_rot = cv2.warpAffine(image, M, (width, height))
    
    image_crop = cv2.getRectSubPix(image_rot, size, center)

    return image_crop

def pointobb2thetaobb_180(pointobb):
    """convert pointobb to thetaobb
    Input:
        pointobb (list[1x8]): [x1, y1, x2, y2, x3, y3, x4, y4]
    Output:
        thetaobb (list[1x5])
    """
    pointobb = np.int0(np.array(pointobb))
    pointobb.resize(4, 2)
    rect = cv2.minAreaRect(pointobb)
    x, y, w, h, theta = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
    theta = theta
    thetaobb = [x, y, w, h, theta]
    
    return thetaobb

def pointobb2bbox(pointobb):
    """
    docstring here
        :param self: 
        :param pointobb: list, [x1, y1, x2, y2, x3, y3, x4, y4]
        return [xmin, ymin, xmax, ymax]
    """
    xmin = min(pointobb[0::2])
    ymin = min(pointobb[1::2])
    xmax = max(pointobb[0::2])
    ymax = max(pointobb[1::2])
    bbox = [xmin, ymin, xmax, ymax]
    
    return bbox

def rotxt_parse(label_file):
    objects = []
    file = open(label_file)
    data = file.readlines()
    for ids, subdata in enumerate(data):
        object_struct = {}
        x1, y1, x2, y2, x3, y3, x4, y4, classname, _ = subdata.split()
        pointobb=[float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4)]
        thetaobb = pointobb2thetaobb_180(pointobb)
        bbox_list = pointobb2bbox(pointobb)
        xmin = bbox_list[0]
        ymin = bbox_list[1]
        xmax = bbox_list[2]
        ymax = bbox_list[3]

        object_struct['pointobb'] = pointobb
        object_struct['rbbox'] = thetaobb
        object_struct['bbox'] = [xmin, ymin, xmax, ymax]
        object_struct['label'] = classname
    
        objects.append(object_struct)
    return objects

 
def save_json(save_path,data):
    assert save_path.split('.')[-1] == 'json'
    with open(save_path,'w') as file:
        json.dump(data,file)

if __name__ == "__main__":
    img_dir = 'PATH_OF_DOTA_IMAGES'
    gt_dir = 'PATH_OF_TXT_GT_FILE'
    save_dir = 'PATH_OF_PATCHES_PNG'
    os.makedirs(save_dir,exist_ok=True)

    file_list = os.listdir(gt_dir)

    cut_patches_json = {'plane':[], 'baseball-diamond':[], 'bridge':[], 'ground-track-field':[],
               'small-vehicle':[], 'large-vehicle':[], 'ship':[], 'tennis-court':[],
               'basketball-court':[], 'storage-tank':[], 'soccer-ball-field':[],
               'roundabout':[], 'harbor':[], 'swimming-pool':[], 'helicopter':[]}

    json_save_path = 'CUT_PATCHES_JSON'


    for idx in trange(len(file_list)):
        gt_filename = file_list[idx]
        filename = gt_filename.split('.txt')[0]
        img_filename = filename + '.png'
        gt_file_path = os.path.join(gt_dir,gt_filename)
        img_file_path = os.path.join(img_dir,img_filename)
        image = cv2.imread(img_file_path)
        objects = rotxt_parse(gt_file_path)
        for ids, obj in enumerate(objects):
            classname = obj['label']
            thetaobb  = obj['rbbox']
            width, height = int(thetaobb[2]), int(thetaobb[3])
            size = width * height
            ratio = max(width/height, height/width)
            cut_box = ((int(thetaobb[0]), int(thetaobb[1])), (width, height), thetaobb[4])
            save_patch_name = classname + '_from_' + filename + str(ids) + '.png'
            save_patch_path = os.path.join(save_dir, save_patch_name)
            cropped = cut_rotated_box(image, cut_box)
            cv2.imwrite(save_patch_path, cropped)
            cut_patches_json[classname].append({'name':save_patch_name,'width':width, 'height':height, 'size':size, 'ratio': ratio})
    
    save_json(json_save_path,cut_patches_json)
        
