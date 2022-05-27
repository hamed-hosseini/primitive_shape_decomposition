import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw
import tensorflow
import keras
from keras import backend as K
from matplotlib import pyplot as plt
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib
import  pandas as pd
import skimage
import general_utils

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
    NAME = "depth_net"
    Network_mode = 'depth' # rgb, depth, rgb_depth
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
    STEPS_PER_EPOCH = 2

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
class CocoLikeDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """

    def load_data(self, annotation_json, images_dir):
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
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    if config.Network_mode == 'rgb':
                        image_file_name = image['file_name']
                    elif config.Network_mode == 'depth':
                        image_file_name = image['file_name'].replace('color', 'depth')
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))

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
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    #hmd
    # IMAGE_MIN_DIM = 480
    # IMAGE_MAX_DIM = 640
    DETECTION_MIN_CONFIDENCE = 0.9

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

    dataset_train = CocoLikeDataset()
    if config.Network_mode =='rgb':
        dataset_train.load_data(os.path.join(os.getcwd(), 'datasets/primitive_shapes/train/coco_annotations.json'),
                                os.path.join(os.getcwd(), 'datasets/primitive_shapes/train/rgb'))
    elif config.Network_mode == 'depth':
        dataset_train.load_data(os.path.join(os.getcwd(), 'datasets/primitive_shapes/train/coco_annotations.json'),
                                os.path.join(os.getcwd(), 'datasets/primitive_shapes/train/depth'))

    dataset_train.prepare()


    dataset_val = CocoLikeDataset()
    if config.Network_mode == 'rgb':
        dataset_val.load_data(os.path.join(os.getcwd(), 'datasets/primitive_shapes/val/coco_annotations.json'),
                              os.path.join(os.getcwd(), 'datasets/primitive_shapes/val/depth'))
    elif config.Network_mode == 'depth':
        dataset_val.load_data(os.path.join(os.getcwd(), 'datasets/primitive_shapes/val/coco_annotations.json'),
                              os.path.join(os.getcwd(), 'datasets/primitive_shapes/val/depth'))
    dataset_val.prepare()

    dataset_test = CocoLikeDataset()
    if config.Network_mode == 'rgb':
        dataset_test.load_data(os.path.join(os.getcwd(), 'datasets/primitive_shapes/test/coco_annotations.json'),
                              os.path.join(os.getcwd(), 'datasets/primitive_shapes/test/rgb'))
    elif config.Network_mode == 'depth':
        dataset_val.load_data(os.path.join(os.getcwd(), 'datasets/primitive_shapes/test/coco_annotations.json'),
                              os.path.join(os.getcwd(), 'datasets/primitive_shapes/test/depth'))
    dataset_test.prepare()

    dataset = dataset_train
    image_ids = np.random.choice(dataset.image_ids, 4)
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

    Train = True
    Test = True
    if Train:
        #...................................Start Trainng...................................................
        # Create model in training mode
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=MODEL_DIR)

        # Which weights to start with?
        init_with = "coco"  # imagenet, coco, or last

        if init_with == "imagenet":
            model.load_weights(model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            model.load_weights(COCO_MODEL_PATH, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                        "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            # Load the last model you trained and continue training
            model.load_weights(model.find_last(), by_name=True)


        ###########ONLY HEADS#########
        # Train the head branches
        # Passing layers="heads" freezes all layers except the head
        # layers. You can also pass a regular expression to select
        # which layers to train by name pattern.
        # start_train = time.time()
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=1,
        #             layers='heads')
        # end_train = time.time()
        # minutes = round((end_train - start_train) / 60, 2)
        # print('Training took {0} minutes'.format(minutes))


        # #########ALL LAYERS#############
        #
        # Fine tune all layers
        # Passing layers="all" trains all layers. You can also
        # pass a regular expression to select which layers to
        # train by name pattern.
        start_train = time.time()
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE * 10,
                    epochs=3,
                    layers="all")
        end_train = time.time()
        minutes = round((end_train - start_train) / 60, 2)
        print('Training took {0} minutes'.format(minutes))
        # #######################################
    if Test:
        #.............................................Finish Trainning...................................................
        inference_config = InferenceConfig()

        # Recreate the model in inference mode
        model = modellib.MaskRCNN(mode="inference",
                                  config=inference_config,
                                  model_dir=MODEL_DIR)

        # Get path to saved weights
        # Either set a specific path or find last trained weights
        # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
        model_path = model.find_last()
        # model_path = os.path.join(MODEL_DIR, "mask_rcnn_cig_butts_0003.h5")

        # Load trained weights (fill in path to trained weights here)
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)

        ious = np.empty((0, config.NUM_CLASSES - 1), float)
        dices = np.empty((0, config.NUM_CLASSES - 1), float)

        my_time = str(time.time())
        os.mkdir(os.path.join('datasets', 'primitive_shapes', 'test', 'rgb', 'predict' + my_time))
        for ind in dataset_test.image_ids:
            if ind == 3:
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
                                        figsize=(15, 15))

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


        # real_test_dir = 'datasets/primitive_shapes/test/rgb'
        # image_paths = []
        # for filename in os.listdir(real_test_dir):
        #     if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        #         image_paths.append(os.path.join(real_test_dir, filename))
        #
        # my_time = str(time.time())
        # os.mkdir(os.path.join(image_paths[0].split('color_image')[0], 'predict' + my_time))
        # ious = []
        # for image_path in image_paths:
        #     img = skimage.io.imread(image_path)
        #     img_arr = np.array(img)
        #     results = model.detect([img_arr], verbose=1)
        #     r = results[0]
        #     mask = r[mask]
        #     mask, class_ids = dataset.load_mask(image_id)
        #     ious.append(iou_coef(results))
        #
        #     visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
        #                                 dataset_val.class_names, r['scores'], title=image_path, my_time=my_time, figsize=(20, 20))