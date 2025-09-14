import os
import random
from tqdm import trange


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

def bbox2pointobb(bbox):
    """
    docstring here
        :param self: 
        :param bbox: list, [xmin, ymin, xmax, ymax]
        return [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    xmin, ymin, xmax, ymax = bbox
    x1, y1 = xmin, ymin
    x2, y2 = xmax, ymin
    x3, y3 = xmax, ymax
    x4, y4 = xmin, ymax

    pointobb = [x1, y1, x2, y2, x3, y3, x4, y4]
    
    return pointobb

def rotxt_parse(label_file):
    objects = []
    file = open(label_file)
    data = file.readlines()
    for ids, subdata in enumerate(data):
        object_struct = {}
        x1, y1, x2, y2, x3, y3, x4, y4, classname, _ = subdata.split()
        pointobb=[float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4)]
        bbox_list = pointobb2bbox(pointobb)
        xmin = bbox_list[0]
        ymin = bbox_list[1]
        xmax = bbox_list[2]
        ymax = bbox_list[3]
        
        bbox_w = xmax - xmin
        bbox_h = ymax - ymin

        object_struct['segmentation'] = pointobb
        object_struct['pointobb'] = (pointobb)
        object_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
        object_struct['label'] = classname
        
        objects.append(object_struct)
    return objects

def count_dataset(label_dir):
    label_list = os.listdir(label_dir)
    instance_num = 0
    img_num = 0
    label_info = {}
    for idx in trange(len(label_list)):
        img_num += 1
        label_file_name = label_list[idx]
        label_file_path = os.path.join(label_dir, label_file_name)
        tree=open(label_file_path)
        root=tree.readlines()
        for single_object in root:
            instance_num = instance_num + 1
            classname = single_object.split(' ')[8]
            if classname in label_info.keys():
                label_info[classname] += 1
            else:
                label_info[classname] = 1
    print('img_num:', img_num)
    print('instance_num:', instance_num)
    print('label_info:', label_info)
    print('num of cls:', len(label_info))
    return instance_num

def parse_cls_list(class_list,K = 1):#
    len_single_img = len(class_list)
    if len_single_img == 1:
        return [0]
    cls_cnt = {}
    for idx,cls in enumerate(class_list):
        if cls not in cls_cnt.keys():
            cls_cnt[cls] = [idx]
        else:
            cls_cnt[cls].append(idx)
    select_idx = []
    for key,val in cls_cnt.items():
        select_num = min(K, len(val))
        select_idx += random.sample(val,select_num)
    return select_idx

root_dir = 'PATH_OF_DOTA_DATASET'

for data_set in ['train']:#,'val'
    annfiles_dir = os.path.join(root_dir,'{}_obb/split_images/annfiles'.format(data_set))
    save_annfiles_dir = os.path.join(root_dir,'{}_obb/split_images/annfiles_{}'.format(data_set,'3ins'))
    os.makedirs(save_annfiles_dir,exist_ok=True)
    ann_list = os.listdir(annfiles_dir)

    for idx in trange(len(ann_list)):
        label_file = ann_list[idx]
        filename = label_file.split('.txt')[0]
        label_file_path = os.path.join(annfiles_dir, label_file)

        save_file_path = os.path.join(save_annfiles_dir, '{}.txt'.format(filename))
        save_txt = open(save_file_path, 'w', encoding='utf-8')

        file = open(label_file_path)
        data = file.readlines()
        if len(data) == 0:
            continue
        class_list = [obj.split()[-2] for obj in data]
        select_idx = parse_cls_list(class_list,K=3)
        for ids in select_idx:
            save_txt.write(data[ids])
    print('Count the dataset path:', annfiles_dir)
    instance_num_before = count_dataset(annfiles_dir)
    print('Count the dataset path:', save_annfiles_dir)
    instance_num_after = count_dataset(save_annfiles_dir)
    
    print('The all datasets have {} instances, we remove {} instances and keep {} instances, take {}\%  proportion.'.format(instance_num_before,(instance_num_before-instance_num_after),instance_num_after,(instance_num_after/instance_num_before*100)))