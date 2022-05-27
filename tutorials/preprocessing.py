import random
import shutil
import numpy as np
import os
from PIL import Image
import json
from my_image import MyImage
import general_utils
if __name__ == '__main__':
    root_dir = "/home/hosseini/Desktop/grasp_primitiveShape/data_generation/logs/2022-04-03.18:16:57/data"
    path = 'datasets/primitive_shapes'
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    if os.path.exists(os.path.join(path, 'train')):
        shutil.rmtree(os.path.join(path, 'train'))
    os.mkdir(os.path.join(path, 'train'))
    if os.path.exists(os.path.join(path, 'train', 'rgb')):
        shutil.rmtree(os.path.join(path, 'train', 'rgb'))
    os.mkdir(os.path.join(path, 'train', 'rgb'))
    if os.path.exists(os.path.join(path, 'train', 'depth')):
        shutil.rmtree(os.path.join(path, 'train', 'depth'))
    os.mkdir(os.path.join(path, 'train', 'depth'))

    if os.path.exists(os.path.join(path, 'val')):
        shutil.rmtree(os.path.join(path, 'val'))
    os.mkdir(os.path.join(path, 'val'))
    if os.path.exists(os.path.join(path, 'val', 'rgb')):
        shutil.rmtree(os.path.join(path, 'val', 'rgb'))
    os.mkdir(os.path.join(path, 'val', 'rgb'))
    if os.path.exists(os.path.join(path, 'val', 'depth')):
        shutil.rmtree(os.path.join(path, 'val', 'depth'))
    os.mkdir(os.path.join(path, 'val', 'depth'))

    if os.path.exists(os.path.join(path, 'test')):
        shutil.rmtree(os.path.join(path, 'test'))
    os.mkdir(os.path.join(path, 'test'))
    if os.path.exists(os.path.join(path, 'test', 'rgb')):
        shutil.rmtree(os.path.join(path, 'test', 'rgb'))
    os.mkdir(os.path.join(path, 'test', 'rgb'))
    if os.path.exists(os.path.join(path, 'test', 'depth')):
        shutil.rmtree(os.path.join(path, 'test', 'depth'))
    os.mkdir(os.path.join(path, 'test', 'depth'))

    train_test_split = 0.7
    train_val_split = 0.9
    all_inds = np.arange(len([name for name in os.listdir(os.path.join(root_dir, 'color_images')) if os.path.isfile(os.path.join(root_dir, 'color_images',name))]))
    random.shuffle(all_inds)
    train_inds = all_inds[:int(train_test_split * train_val_split * len(all_inds))]
    valid_inds = all_inds[int(train_test_split * train_val_split * len(all_inds)):int(train_test_split * len(all_inds))]
    test_inds = all_inds[int(train_test_split * len(all_inds)):]
    # stages = [('train',train_inds), (valid_inds, 'valid'), (test_inds, 'test')]
    # for stage, li in stages:
    desc_images_train = []
    desc_images_val = []
    desc_images_test = []

    desc_annotations_train = []
    desc_annotations_val = []
    desc_annotations_test = []

    anno_json_train = {}
    anno_json_val = {}
    anno_json_test = {}

    ann_ind_start_train = 0
    ann_ind_start_val = 0
    ann_ind_start_test = 0
    for root, directories, files in os.walk(os.path.join(root_dir, 'color_images'), topdown=False):
        for ind, name in enumerate(sorted(files)):
            if ind%10 == 0:
                print(ind)
            im_mask = Image.open(os.path.join(root, name)).convert('RGB')
            im_depth = Image.open(os.path.join(
                os.path.join(root_dir, 'depth_images'),
                'depth_image_' + name.split('.png')[0].split('_')[2] + '.png'))
            f_info = open((os.path.join(os.path.join(root_dir, 'info'), 'info_image_' + name.split('.png')[0].split('_')[2]) + '.json'))
            info = json.load(f_info)
            f_info.close()
            if ind in train_inds:
                ann_ind_start = ann_ind_start_train
            elif ind in valid_inds:
                ann_ind_start = ann_ind_start_val
            elif ind in test_inds:
                ann_ind_start = ann_ind_start_test
            image = MyImage(name=name, id=ind, rgb=im_mask, mask=im_mask, depth=im_depth, info=info, ann_ind_start=ann_ind_start, is_crowd=0)
            image.recolor()
            if ind in train_inds:
                desc_images_train.append(image.get_desc_image())
                desc_annotations_train.extend(image.get_desc_anno())
                ann_ind_start_train += image.count_ann
                image.imsave(path, 'train')
            elif ind in valid_inds:
                desc_images_val.append(image.get_desc_image())
                desc_annotations_val.extend(image.get_desc_anno())
                ann_ind_start_val += image.count_ann
                image.imsave(path, 'val')
            elif ind in test_inds:
                desc_images_test.append(image.get_desc_image())
                desc_annotations_test.extend(image.get_desc_anno())
                ann_ind_start_test += image.count_ann
                image.imsave(path, 'test')
    anno_json_train['images'] = desc_images_train
    anno_json_val['images'] = desc_images_val
    anno_json_test['images'] = desc_images_test

    anno_json_train['info'] = {
        "description": "Primitive Shape Decomposition Dataset",
        "url": "www.google.com",
        "version": "1.0",
        "year": 2021,
        "contributor": "Hamed Hosseini",
        "date_created": "2021/09/01"
    }
    anno_json_val['info'] = {
        "description": "Primitive Shape Decomposition Dataset",
        "url": "www.google.com",
        "version": "1.0",
        "year": 2021,
        "contributor": "Hamed Hosseini",
        "date_created": "2021/09/01"
    }
    anno_json_test['test'] = {
        "description": "Primitive Shape Decomposition Dataset",
        "url": "www.google.com",
        "version": "1.0",
        "year": 2021,
        "contributor": "Hamed Hosseini",
        "date_created": "2021/09/01"
    }


    anno_json_train['categories'] = [{"supercategory": "primitive_shape", "id": key, "name": value }
                                     for key, value in general_utils.id_category.items()]
    anno_json_val['categories'] = [{"supercategory": "primitive_shape", "id": key, "name": value }
                                   for key, value in general_utils.id_category.items()]
    anno_json_test['categories'] = [{"supercategory": "primitive_shape", "id": key, "name": value }
                                    for key, value in general_utils.id_category.items()]

    anno_json_train['annotations'] = desc_annotations_train
    anno_json_val['annotations'] = desc_annotations_val
    anno_json_test['annotations'] = desc_annotations_test

    with open(os.path.join(path, 'train', 'coco_annotations.json'), 'w') as f:
        json.dump(anno_json_train, f)

    with open(os.path.join(path,'val','coco_annotations.json'), 'w') as f:
            json.dump(anno_json_val, f)

    with open(os.path.join(path,'test', 'coco_annotations.json'), 'w') as f:
            json.dump(anno_json_test, f)
