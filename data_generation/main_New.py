#!/usr/bin/env python

import time
import os
import random
import shutil
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2
from collections import namedtuple
from robot_New import Robot
from logger import Logger
import utils

import glob

def create_objects_from_shape(shapes,obj_mesh_dir):
    """ Create objects folder for simulation from shapes generated"""
    # delete and create block_gen directory for the simulator
    if os.path.exists(obj_mesh_dir):
        shutil.rmtree(obj_mesh_dir)
    os.mkdir(obj_mesh_dir)

    obj_shape_dir = os.path.abspath("objects/primitive_shapes")

    obj_list=[]
    for iter, shape in enumerate(shapes):
        # sample one object from shape primitive directory
        obj_source = random.choice(os.listdir(os.path.join(obj_shape_dir, shape + "_obj/")))
        if obj_source in obj_list:
            obj_dest = obj_source.split('.obj')[0]+'1'+'.obj'
        else:
            obj_dest = obj_source
        obj_list.append(obj_dest)
        shutil.copy(os.path.join(obj_shape_dir, shape + "_obj/", obj_source), os.path.join(obj_mesh_dir, obj_dest))
    return obj_list

def main(args, shapes):
    is_sim = True # simulation

    max_num_obj = len(shapes) # number of object

    port_num=int(args.port)
    obj_mesh_dir = "objects/block_gen_" + str(port_num)  # where .obj are located
    obj_mesh_dir = os.path.abspath(obj_mesh_dir)

    # workspace in simulation
    workspace_limits = np.asarray([[0.275, 0.53], [-0.235, 0.19], [-0.1501, 0.3]])
    heightmap_resolution = 0.002 # metertqdm
    random_seed = 1234
    np.random.seed(random_seed)

    ## Pre-loading and logging options
    continue_logging = False # Continue logging from previous session
    logging_directory = os.path.abspath('logs')


    ## Initialize data logger
    logger = Logger(continue_logging, logging_directory, max_num_obj, offset=int(args.offset))
    ## Start main training/testing loop

    NUM_ITERATIONS = int(args.num_iterations)
    for iter in range(NUM_ITERATIONS):
        #hmd
        random.shuffle(shapes)
        shapes_sampled = random.sample(shapes, random.randint(1,8))
        random.shuffle(shapes_sampled)
        num_obj = len(shapes_sampled)
        #hmd
        obj_list = create_objects_from_shape(shapes_sampled, obj_mesh_dir)
        # Save obj list
        logger.save_obj_list(iter, obj_list)

        # Initialize pick-and-place system (camera and robot)
        robot = Robot(is_sim, obj_mesh_dir, num_obj, shapes_sampled, workspace_limits, port_num)
        print('hi')
        # Get latest RGB-D image
        color_img, depth_img = robot.get_camera_data()
        depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration

        if (depth_img == 0).all():
                continue

        # Save RGB-D images 
        logger.save_images(iter, color_img, depth_img, '0')
        # logger.save_heightmaps(iter, color_heightmap, valid_depth_heightmap)
        shape_color_json = robot.get_shape_color_json()
        logger.save_json(iter, shape_color_json)

        ## iterative solution for getting modal segmasks using opencv and not needing cameras in Vrep
        thresh = [[0.300, 0.470, 0.650],  # blue
                  [0.340, 0.630, 0.300],  # green
                  [0.610, 0.450, 0.370],  # brown
                  [0.940, 0.550, 0.160],  # orange
                  [0.920, 0.780, 0.280],  # yellow
                  [0.749, 0.937, 0.270],  # lime
                  [0.660, 1.000, 0.760],  # mint
                  [0.690, 0.470, 0.631],  # purple
                  [0.462, 0.717, 0.698],  # cyan
                  [0.990, 0.615, 0.654]]  #pink
        segmodal_img_black_white = np.zeros((480, 640, 3)).astype(np.uint8)
        segmodal_img_bin = np.zeros((480, 640, 3)).astype(np.uint8)
        img = color_img / 255
        for idx_obj in range(num_obj):
            cur_thresh = np.array(thresh[idx_obj])
            margin = 0.1
            lower_color = cur_thresh * (1.0 - margin)
            upper_color = cur_thresh * (1.0 + margin)

            mask = cv2.inRange(img, lower_color, upper_color)
            res = cv2.bitwise_and(img, img, mask=mask) # keep the values where mask = 1
            res = np.where(res != 0, 1, 0) # res !=0 then 1 else 0

            # construct segmodal black and white image
            segmodal_img_black_white = segmodal_img_black_white + res * 255
            segmodal_img_black_white = np.minimum(segmodal_img_black_white, 255)
            segmodal_img_black_white = segmodal_img_black_white.astype(np.uint8)
            # construct segmodal bin image
            bicolor_img_mask = np.where(res != 0, (idx_obj + 1), 0).astype(np.uint8)
            segmodal_img_bin = segmodal_img_bin + bicolor_img_mask


        logger.save_segmask(iter, segmodal_img_black_white, segmodal_img_bin)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data Collection Process")
    parser.add_argument("-n", "--num_iterations", default=300, help="number of images to collect")
    parser.add_argument("-o", "--offset", default=0, help="start saving images starting from #offset image")
    parser.add_argument("-p", "--port", default=19997, help="port number")
    args = parser.parse_args()
    # Shapes to use
    shapes = ["Semisphere","Semisphere","Cuboid", "Cuboid", "Cylinder","Cylinder", "Ring","Ring", "Stick","Stick", "Sphere", "Sphere"]
    # shapes = ["Cylinder", "Cuboid", "Ring", "Stick", "Sphere", "Semisphere"]
    main(args, shapes)
