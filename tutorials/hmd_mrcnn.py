import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw
import tensorflow as tf
import keras
from keras import backend as K
from matplotlib import pyplot as plt
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib
import pandas as pd
import skimage
import general_utils

## ino chk konid.....
import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")
def scheduler(epoch):
    if epoch < config.EPOCHS / 3:
        return config.LEARNING_RATE
    elif epoch < 2 * config.EPOCHS / 3:
        return config.LEARNING_RATE * 0.1
    else:
        return config.LEARNING_RATE * 0.01

def dice_coef(y_true, y_pred, smooth=1):
    intersection = np.sum(y_true * y_pred, axis=(0, 1))
    union = np.sum(y_true, axis=(0, 1)) + np.sum(y_pred, axis=(0, 1))
    # dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def iou_coef(y_true, y_pred, smooth=1):
  intersection = np.sum(y_true * y_pred, axis=(0, 1))
  union = np.sum(y_true, axis=(0, 1)) + np.sum(y_pred, axis=(0, 1)) - intersection
  # iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  iou = (intersection + smooth) / (union + smooth)
  return iou

class CigButtsConfig(Config):
    """Configuration for training on the cigarette butts dataset.
    Derives from the base Config class and overrides values specific
    to the cigarette butts dataset.
    """
    # Give the configuration a recognizable name
    NAME = "rgbd_scaleupd1000_datasets_v1"
    dataset_name = 'datasets_v1'
    Train = True
    # Train = False
    Test = True
    # Test = False
    # debug = True
    debug = False
    train_mode = 'all' # transfer or all
    # train_mode = 'transfer'
    Network_mode = 'depth' # rgb, depth, rgb_depth
    # Network_mode = 'rgb' # rgb, depth, rgb_depth, gray
    # Network_mode = 'gray' # rgb, depth, rgb_depth, gray
    Network_mode = 'rgb_depth' # rgb, depth, rgb_depth
    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    # NUM_CLASSES = 1 + 1  # background + 1 (cig_butt)
    #hmd
    NUM_CLASSES = 1 + 9  # background + 9 (primitive shape)

    # All of our training images are 512x512
    # IMAGE_MIN_DIM = 512
    # IMAGE_MAX_DIM = 512
    #hmd
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 50
    # STEPS_PER_EPOCH = 3
    EPOCHS = 100
    # EPOCHS = 3

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 5

    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = 'resnet101'

    # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50
    POST_NMS_ROIS_INFERENCE = 500
    POST_NMS_ROIS_TRAINING = 1000
    USE_MINI_MASK = False
    compute_mean_pixel_size = True
    # compute_mean_pixel_size = False
    if Network_mode == 'rgb':
        IMAGE_CHANNEL_COUNT = 3
        MEAN_PIXEL = np.array([254.1749290922619, 254.1700030810805, 253.69742554253475])
    if Network_mode == 'gray':
        IMAGE_CHANNEL_COUNT = 1
        MEAN_PIXEL = np.array([0.500000])
    if Network_mode == 'depth':
        IMAGE_CHANNEL_COUNT = 1
        MEAN_PIXEL = np.array([0.12916169411272974])
    if Network_mode == 'rgb_depth':
        IMAGE_CHANNEL_COUNT = 4
        MEAN_PIXEL = np.array([254.1749290922619, 254.1700030810805, 253.69742554253475, 0.12916169411272974])
    MIN_PIXEL = 0
    MAX_PIXEL = 0
    STD_PIXEL = 1
class CocoLikeDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """

    def load_data(self, annotation_json, images_dir, mean_pixel_evaluate=False):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()

        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(
                    class_name))
                return

            self.add_class(source_name, class_id, class_name)

        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        # Get all images and add them to the dataset
        seen_images = {}
        mean_R_s, mean_G_s, mean_B_s, mean_D_s, mean_grays = [], [], [], [], []
        min_D_s ,max_D_s = [], []
        std_R_s, std_G_s, std_B_s, std_D_s = [], [], [], []
        for indexx, image in enumerate(coco_json['images']):
            if indexx % 500 == 0:
                print(indexx)
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    if config.Network_mode == 'rgb' or config.Network_mode == 'rgb_depth' or config.Network_mode == 'gray':
                        image_file_name = image['file_name']
                    elif config.Network_mode == 'depth':
                        image_file_name = image['file_name'].replace('color', 'depth')
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                if config.Network_mode == 'rgb_depth':
                    image_path = os.path.abspath(os.path.join(images_dir, 'rgb', image_file_name)), \
                                 os.path.abspath(os.path.join(images_dir, 'depth', image_file_name.replace('color_image', 'depth_image')))
                else:
                    image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]

                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )
                if mean_pixel_evaluate:
                    my_image = self.load_image(-1)
                    original_shape = my_image.shape
                    my_image, window, scale, padding, crop = utils.resize_image(
                        my_image,
                        min_dim=config.IMAGE_MIN_DIM,
                        min_scale=config.IMAGE_MIN_SCALE,
                        max_dim=config.IMAGE_MAX_DIM,
                        mode=config.IMAGE_RESIZE_MODE)


                    if config.Network_mode == 'rgb' or config.Network_mode == 'rgb_depth' :
                        mean_R_s += [my_image.mean(axis=(0, 1))[0]]
                        mean_G_s += [my_image.mean(axis=(0, 1))[1]]
                        mean_B_s += [my_image.mean(axis=(0, 1))[2]]

                        std_R_s += [my_image.std(axis=(0, 1))[0]]
                        std_G_s += [my_image.std(axis=(0, 1))[1]]
                        std_B_s += [my_image.std(axis=(0, 1))[2]]
                    if  config.Network_mode == 'rgb_depth':
                        mean_D_s += [my_image.mean(axis=(0, 1))[3]]
                        min_D_s += [my_image.min(axis=(0, 1))[3]]
                        max_D_s += [my_image.max(axis=(0, 1))[3]]
                        std_D_s += [my_image.std(axis=(0, 1))[3]]
                    if config.Network_mode=='depth':
                        mean_D_s += [my_image.mean(axis=(0, 1))[0]]
                        min_D_s += [my_image.min(axis=(0, 1))[0]]
                        max_D_s += [my_image.max(axis=(0, 1))[0]]
                        std_D_s += [my_image.std(axis=(0, 1))[0]]
                    if config.Network_mode == 'gray':
                        mean_grays += [my_image.mean(axis=(0, 1))[0]]
        if mean_pixel_evaluate:
            if config.Network_mode == 'rgb':
                config.MEAN_PIXEL = np.array([np.mean(mean_R_s), np.mean(mean_G_s), np.mean(mean_B_s)])
                config.MIN_PIXEL = np.array([0, 0, 0])
                config.MAX_PIXEL = np.array([255, 255, 255])
                config.STD_PIXEL = np.array([np.std(std_R_s), np.std(std_G_s), np.std(std_B_s)])
            elif config.Network_mode == 'gray':
                config.MEAN_PIXEL = np.array([np.mean(mean_grays)])
                config.MIN_PIXEL = np.array([0])
                config.MAX_PIXEL = np.array([255])
                #should modify since it was useless didnt modified
                config.STD_PIXEL = np.array([np.std(std_R_s)])
            elif config.Network_mode == 'rgb_depth':
                config.MEAN_PIXEL = np.array([np.mean(mean_R_s), np.mean(mean_G_s), np.mean(mean_B_s), np.mean(mean_D_s)])
                config.MIN_PIXEL = np.array([0, 0, 0, np.min(min_D_s)])
                config.MAX_PIXEL = np.array([255, 255, 255, np.max(max_D_s)])
                config.STD_PIXEL = np.array([np.std(std_R_s), np.std(std_G_s), np.std(std_B_s), np.std(std_D_s)])
            elif config.Network_mode == 'depth':
                config.MEAN_PIXEL = np.array([np.mean(mean_D_s)])
                config.MIN_PIXEL = np.array([np.min(min_D_s)])
                config.MAX_PIXEL = np.array([np.max(max_D_s)])
                config.STD_PIXEL = np.array([np.std(std_D_s)])
            # hmd: saving essential configes
            with open('conf.json', 'w') as f:
                my_dict = {}
                my_dict['MEAN_PIXEL'] = config.MEAN_PIXEL.tolist()
                # my_dict['MIN_PIXEL'] = config.MIN_PIXEL.tolist()
                # my_dict['MAX_PIXEL'] = config.MAX_PIXEL.tolist()
                # my_dict['STD_PIXEL'] = config.STD_PIXEL.tolist()
                json.dump(my_dict, f)
        else:
            with open('conf.json', 'r') as f:
                my_dict = json.load(f)
                if config.Network_mode == 'rgb':
                    # config.MIN_PIXEL = np.array(my_dict['MIN_PIXEL'])
                    # config.MAX_PIXEL = np.array(my_dict['MAX_PIXEL'])
                    config.MEAN_PIXEL = np.array(my_dict['MEAN_PIXEL'])
                    # config.STD_PIXEL = np.array(my_dict['STD_PIXEL'])
                elif config.Network_mode == 'gray':
                    # config.MEAN_PIXEL = np.array([190.63119681919645])
                    config.MEAN_PIXEL = np.array(my_dict['MEAN_PIXEL'])
                elif config.Network_mode == 'depth':
                    # config.MIN_PIXEL = np.array(my_dict['MIN_PIXEL'])
                    # config.MAX_PIXEL = np.array(my_dict['MAX_PIXEL'])
                    config.MEAN_PIXEL = np.array(my_dict['MEAN_PIXEL'])
                    # config.STD_PIXEL = np.array(my_dict['STD_PIXEL'])
                elif config.Network_mode == 'rgb_depth':
                    # config.MIN_PIXEL = np.array(my_dict['MIN_PIXEL'])
                    # config.MAX_PIXEL = np.array(my_dict['MAX_PIXEL'])
                    config.MEAN_PIXEL = np.array(my_dict['MEAN_PIXEL'])
                    # config.STD_PIXEL = np.array(my_dict['STD_PIXEL'])


    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids


class InferenceConfig(CigButtsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # IMAGE_MIN_DIM = 512
    # IMAGE_MAX_DIM = 512
    #hmd
    # IMAGE_MIN_DIM = 480
    # IMAGE_MAX_DIM = 640
    DETECTION_MIN_CONFIDENCE = 0.8
    load_model = 'last' # model_name, last
    # load_model = 'gray_one_channel_datasets_v120221010T1317/mask_rcnn_gray_one_channel_datasets_v1_0099.h5' # model_name, last

if __name__=='__main__':
    print(os.getcwd())
    # Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
    ROOT_DIR = './'
    assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist. Did you forget to read the instructions above? ;)'
    # Import mrcnn libraries
    sys.path.append(ROOT_DIR)


    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)
    config = CigButtsConfig()
    config.display()



    # image_ids = np.random.choice(dataset_.image_ids, 4)
    # for image_id in image_ids:
    #     image = dataset.load_image(image_id)
    #     mask, class_ids = dataset.load_mask(image_id)
    #     visualize.display_top_masks(image, mask, class_ids, dataset.class_names)


    if config.Train:
        dataset_train = CocoLikeDataset(network_mode=config.Network_mode)
        if config.debug:
            if config.Network_mode == 'rgb' or config.Network_mode == 'gray' :
                dataset_train.load_data(
                    os.path.join(os.getcwd(), 'datasets_debug/primitive_shapes/train/coco_annotations.json'),
                    os.path.join(os.getcwd(), 'datasets_debug/primitive_shapes/train/rgb'), mean_pixel_evaluate=config.compute_mean_pixel_size)
            elif config.Network_mode == 'depth':
                dataset_train.load_data(
                    os.path.join(os.getcwd(), 'datasets_debug/primitive_shapes/train/coco_annotations.json'),
                    os.path.join(os.getcwd(), 'datasets_debug/primitive_shapes/train/depth'), mean_pixel_evaluate=config.compute_mean_pixel_size)
            elif config.Network_mode =='rgb_depth':
                dataset_train.load_data(
                    os.path.join(os.getcwd(), 'datasets_debug/primitive_shapes/train/coco_annotations.json'),
                    os.path.join(os.getcwd(), 'datasets_debug/primitive_shapes/train'), mean_pixel_evaluate=config.compute_mean_pixel_size)
        else:
            if config.Network_mode == 'rgb' or config.Network_mode == 'gray' :
                dataset_train.load_data(
                    os.path.join(os.getcwd(), config.dataset_name + '/primitive_shapes/train/coco_annotations.json'),
                    os.path.join(os.getcwd(), config.dataset_name + '/primitive_shapes/train/rgb'), mean_pixel_evaluate=config.compute_mean_pixel_size)
            elif config.Network_mode == 'depth':
                dataset_train.load_data(
                    os.path.join(os.getcwd(), config.dataset_name + '/primitive_shapes/train/coco_annotations.json'),
                    os.path.join(os.getcwd(), config.dataset_name + '/primitive_shapes/train/depth'), mean_pixel_evaluate=config.compute_mean_pixel_size)
            elif config.Network_mode =='rgb_depth':
                dataset_train.load_data(
                    os.path.join(os.getcwd(), config.dataset_name + '/primitive_shapes/train/coco_annotations.json'),
                    os.path.join(os.getcwd(), config.dataset_name + '/primitive_shapes/train'), mean_pixel_evaluate=config.compute_mean_pixel_size)
        dataset_train.prepare()
        dataset_val = CocoLikeDataset(network_mode=config.Network_mode)
        if config.debug:
            if config.Network_mode == 'rgb' or config.Network_mode == 'gray' :
                dataset_val.load_data(
                    os.path.join(os.getcwd(), 'datasets_debug/primitive_shapes/val/coco_annotations.json'),
                    os.path.join(os.getcwd(), 'datasets_debug/primitive_shapes/val/rgb'))
            elif config.Network_mode == 'depth':
                dataset_val.load_data(
                    os.path.join(os.getcwd(), 'datasets_debug/primitive_shapes/val/coco_annotations.json'),
                    os.path.join(os.getcwd(), 'datasets_debug/primitive_shapes/val/depth'))
            elif config.Network_mode == 'rgb_depth':
                dataset_val.load_data(
                    os.path.join(os.getcwd(), 'datasets_debug/primitive_shapes/val/coco_annotations.json'),
                    os.path.join(os.getcwd(), 'datasets_debug/primitive_shapes/val'))
        else:
            if config.Network_mode == 'rgb' or config.Network_mode == 'gray' :
                dataset_val.load_data(
                    os.path.join(os.getcwd(), config.dataset_name + '/primitive_shapes/val/coco_annotations.json'),
                    os.path.join(os.getcwd(), config.dataset_name + '/primitive_shapes/val/rgb'))
            elif config.Network_mode == 'depth':
                dataset_val.load_data(
                    os.path.join(os.getcwd(), config.dataset_name + '/primitive_shapes/val/coco_annotations.json'),
                    os.path.join(os.getcwd(), config.dataset_name + '/primitive_shapes/val/depth'))
            elif config.Network_mode == 'rgb_depth':
                dataset_val.load_data(
                    os.path.join(os.getcwd(), config.dataset_name + '/primitive_shapes/val/coco_annotations.json'),
                    os.path.join(os.getcwd(), config.dataset_name + '/primitive_shapes/val'))

        dataset_val.prepare()

        #...................................Start Trainng...................................................
        # Create model in training mode
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=MODEL_DIR)


        # Which weights to start with?
        init_with = "coco"  # imagenet, coco, or last or nothing
        # init_with = "last"  # imagenet, coco, or last or nothing
        # init_with = "nothing"  # imagenet, coco, or last or nothing

        if init_with == "imagenet":
            model.load_weights(model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            if config.Network_mode == 'rgb' :
                model.load_weights(COCO_MODEL_PATH, by_name=True,
                                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                            "mrcnn_bbox", "mrcnn_mask"])
            elif config.Network_mode == 'depth' or config.Network_mode == 'rgb_depth'or config.Network_mode == 'gray' :
                model.load_weights(COCO_MODEL_PATH, by_name=True,
                                   exclude=["conv1", "mrcnn_class_logits", "mrcnn_bbox_fc",
                                            "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            # Load the last model you trained and continue training
            model.load_weights(model.find_last(), by_name=True)

        callback = keras.callbacks.LearningRateScheduler(scheduler)
        if config.train_mode == 'transfer':
            ###########ONLY HEADS#########
            # Train the head branches
            # Passing layers="heads" freezes all layers except the head
            # layers. You can also pass a regular expression to select
            # which layers to train by name pattern.
            start_train = time.time()
            model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,
                        epochs=config.EPOCHS,
                        layers='heads', custom_callbacks=[callback])
            end_train = time.time()
            minutes = round((end_train - start_train) / 60, 2)
            print('Training took {0} minutes'.format(minutes))

        elif config.train_mode == 'all':
            # #########ALL LAYERS#############
            #
            # Fine tune all layers
            # Passing layers="all" trains all layers. You can also
            # pass a regular expression to select which layers to
            # train by name pattern.
            start_train = time.time()
            model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,
                        epochs=config.EPOCHS,
                        layers="all", custom_callbacks=[callback])
            end_train = time.time()
            minutes = round((end_train - start_train) / 60, 2)
            print('Training took {0} minutes'.format(minutes))


    #.............................................Finish Trainning...................................................
    if config.Test:
        print('Testing')

        dataset_test = CocoLikeDataset(network_mode=config.Network_mode)
        if config.debug:
            if config.Network_mode == 'rgb' or config.Network_mode == 'gray':
                dataset_test.load_data(
                    os.path.join(os.getcwd(), 'datasets_debug/primitive_shapes/test/coco_annotations.json'),
                    os.path.join(os.getcwd(), 'datasets_debug/primitive_shapes/test/rgb'))
            elif config.Network_mode == 'depth':
                dataset_test.load_data(
                    os.path.join(os.getcwd(), 'datasets_debug/primitive_shapes/test/coco_annotations.json'),
                    os.path.join(os.getcwd(), 'datasets_debug/primitive_shapes/test/depth'))
            elif config.Network_mode == 'rgb_depth':
                dataset_test.load_data(
                    os.path.join(os.getcwd(), 'datasets_debug/primitive_shapes/test/coco_annotations.json'),
                    os.path.join(os.getcwd(), 'datasets_debug/primitive_shapes/test'))
        else:
            if config.Network_mode == 'rgb' or config.Network_mode == 'gray':
                dataset_test.load_data(
                    os.path.join(os.getcwd(), config.dataset_name + '/primitive_shapes/test/coco_annotations.json'),
                    os.path.join(os.getcwd(), config.dataset_name + '/primitive_shapes/test/rgb'))
            elif config.Network_mode == 'depth':
                dataset_test.load_data(
                    os.path.join(os.getcwd(), config.dataset_name + '/primitive_shapes/test/coco_annotations.json'),
                    os.path.join(os.getcwd(), config.dataset_name + '/primitive_shapes/test/depth'))
            elif config.Network_mode == 'rgb_depth':
                dataset_test.load_data(
                    os.path.join(os.getcwd(), config.dataset_name + '/primitive_shapes/test/coco_annotations.json'),
                    os.path.join(os.getcwd(), config.dataset_name + '/primitive_shapes/test'))
        dataset_test.prepare()

        inference_config = InferenceConfig()
        with open('conf.json', 'r') as f:
            my_dict = json.load(f)
            if config.Network_mode == 'rgb_depth':
                # inference_config.MIN_PIXEL = np.array(my_dict['MIN_PIXEL'])
                # inference_config.MAX_PIXEL = np.array(my_dict['MAX_PIXEL'])
                inference_config.MEAN_PIXEL = np.array(my_dict['MEAN_PIXEL'])
                # inference_config.STD_PIXEL = np.array(my_dict['STD_PIXEL'])
            elif config.Network_mode == 'depth':
                # inference_config.MIN_PIXEL = np.array(my_dict['MIN_PIXEL'])
                # inference_config.MAX_PIXEL = np.array(my_dict['MAX_PIXEL'])
                inference_config.MEAN_PIXEL = np.array(my_dict['MEAN_PIXEL'])
                # inference_config.STD_PIXEL = np.array(my_dict['STD_PIXEL'])
            elif config.Network_mode == 'rgb':
                # inference_config.MIN_PIXEL = np.array(my_dict['MIN_PIXEL'])
                # inference_config.MAX_PIXEL = np.array(my_dict['MAX_PIXEL'])
                inference_config.MEAN_PIXEL = np.array(my_dict['MEAN_PIXEL'])
                # inference_config.STD_PIXEL = np.array(my_dict['STD_PIXEL'])
            elif config.Network_mode == 'gray':
                # inference_config.MEAN_PIXEL =  np.array([190.63119681919645])
                inference_config.MEAN_PIXEL = np.array(my_dict['MEAN_PIXEL'])
        # Recreate the model in inference mode
        model = modellib.MaskRCNN(mode="inference",
                                  config=inference_config,
                                  model_dir=MODEL_DIR)

        # Get path to saved weights
        # Either set a specific path or find last trained weights
        # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
        if InferenceConfig.load_model == 'last':
            model_path = model.find_last()
        else:
            model_path = os.path.join(MODEL_DIR, InferenceConfig.load_model)

        # Load trained weights (fill in path to trained weights here)
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)

        ious = np.empty((0, config.NUM_CLASSES - 1), float)
        dices = np.empty((0, config.NUM_CLASSES - 1), float)

        my_time = str(time.time())
        if config.debug:
            if config.Network_mode == 'rgb' or config.Network_mode == 'rgb_depth' or config.Network_mode == 'gray':
                os.mkdir(os.path.join('datasets_debug', 'primitive_shapes', 'test', 'rgb', 'predict' + my_time))
            elif config.Network_mode == 'depth':
                os.mkdir(os.path.join('datasets_debug', 'primitive_shapes', 'test', 'depth', 'predict' + my_time))
        else:
            if config.Network_mode == 'rgb' or config.Network_mode == 'rgb_depth' or config.Network_mode == 'gray':
                os.mkdir(os.path.join(config.dataset_name, 'primitive_shapes', 'test', 'rgb', 'predict' + my_time))
            elif config.Network_mode == 'depth':
                os.mkdir(os.path.join(config.dataset_name, 'primitive_shapes', 'test', 'depth', 'predict' + my_time))
        for ind in dataset_test.image_ids:
            if config.debug:
                if ind == 4:
                    break
            print(ind, ' from: ', len(dataset_test.image_ids))
            img = dataset_test.load_image(ind)
            img_arr = np.array(img)
            results = model.detect([img_arr], verbose=0)
            r = results[0]
            mask_pred = r['masks']
            class_ids_pred = r['class_ids']

            mask_target, class_ids_target = dataset_test.load_mask(ind)
            mask_pred_shape = list(mask_pred.shape)
            mask_pred_shape[-1] = config.NUM_CLASSES - 1  # Ommiting background-- because the prediction doesnt have it!
            mask_target_shape = list(mask_target.shape)
            mask_target_shape[-1] = config.NUM_CLASSES - 1  # Ommiting background-- because the prediction doesnt have it!

            mask_pred_final = np.full(mask_pred_shape, False, dtype=bool, order='C')
            mask_target_final = np.full(mask_target_shape, False, dtype=bool, order='C')
            for index, class_id in enumerate(class_ids_pred):
                mask_pred_final[:, :, class_id] += mask_pred[:, :, index]
            for index, class_id in enumerate(class_ids_target):
                mask_target_final[:, :, class_id] += mask_target[:, :, index]
            visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                        dataset_test.class_names, r['scores'], title=ind, my_time=my_time,
                                        figsize=(15, 15),debug=config.debug, config=config)

            ious = np.append(ious, np.array([iou_coef(mask_target_final, mask_pred_final, smooth=0)]), axis=0)
            dices = np.append(dices, np.array([dice_coef(mask_target_final, mask_pred_final, smooth=0)]), axis=0)

        print('IoUs')
        iou_total = []
        for ind, col in enumerate(ious.T):
            iou_class = np.mean([i for i in col if not pd.isna(i)])
            if not pd.isna(iou_class):
                iou_total = iou_total + [iou_class]
                print(general_utils.id_category[ind], iou_class)
        print('mean IoU:', np.mean(iou_total))

        print('Dices')
        dice_total = []
        for ind, col in enumerate(dices.T):
            dice_class = np.mean([i for i in col if not pd.isna(i)])
            if not pd.isna(dice_class):
                dice_total += [dice_class]
                print(general_utils.id_category[ind], dice_class)
        print('mean Dice:', np.mean(dice_total))
