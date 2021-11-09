import os
import random
import sys
sys.path.append(os.path.abspath("/home/hamed/Desktop/grasp_primitiveShape"))
from config import color_space
from PIL import Image
from skimage import measure  # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon  # (pip install Shapely)
import numpy as np
from matplotlib import pyplot as plt


class MyImage():
    def __init__(self, name, id,  rgb, mask, depth, info, ann_ind_start, is_crowd):
        self.name = name
        self.id = id
        self.rgb = rgb
        self.mask = mask
        self.depth = depth
        self.info = info
        self.width, self.height = rgb.size
        self.ann_ind_start = ann_ind_start
        self.is_crowd = is_crowd
        self.annotations = []
        self.primitives = {'cuboid':1, 'sphere':2, 'semisphere':3, 'cylinder':4, 'ring':5, 'stick':6}

    def recolor(self):
        origin_colors = [eval(i) for i in self.info.keys()]
        transfored_colors = random.choices(color_space, k=len(origin_colors))
        random.shuffle(transfored_colors)
        data = np.array(self.rgb)
        for key, color in enumerate(origin_colors):
            r1, g1, b1 = color[0] * 255.0, color[1] * 255.0, color[2] * 255.0   # Original value
            r2, g2, b2 = transfored_colors[key][0] * 255.0, transfored_colors[key][1] * 255.0, transfored_colors[key][2] * 255.0   # Value that we want to replace it with
            red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
            mask = (red == r1) & (green == g1) & (blue == b1)
            data[:, :, :3][mask] = [r2, g2, b2]
        self.rgb = Image.fromarray(data, 'RGB')
        # self.rgb.save(self.name)
    def get_desc_image(self):
        my_dict = {}
        my_dict['file_name'] = self.name
        my_dict['height'], my_dict['width'] = self.height, self.width
        my_dict['id'] = self.id
        return my_dict

    def get_desc_anno(self):
        self.create_sub_masks()
        for ind, (color, sub_mask) in enumerate(self.sub_masks.items()):
            normalized_color = str(tuple(np.asarray(eval(color))/255))
            category_id = self.primitives[self.info[normalized_color]]
            annotation_id = self.ann_ind_start + ind
            self.annotations.append(self.create_sub_mask_annotation(sub_mask, category_id, annotation_id, 0))
            # print('file_name', files[ind])
        return self.annotations

    def create_sub_masks(self):
        # Initialize a dictionary of sub-masks indexed by RGB colors
        self.sub_masks = {}
        for x in range(self.width):
            for y in range(self.height):
                # Get the RGB values of the pixel
                pixel = self.mask.getpixel((x, y))[:3]

                # If the pixel is not black...
                if pixel != (255, 255, 255):
                    # Check to see if we've created a sub-mask...
                    pixel_str = str(pixel)
                    sub_mask = self.sub_masks.get(pixel_str)
                    if sub_mask is None:
                        # Create a sub-mask (one bit per pixel) and add to the dictionary
                        # Note: we add 1 pixel of padding in each direction
                        # because the contours module doesn't handle cases
                        # where pixels bleed to the edge of the image
                        self.sub_masks[pixel_str] = Image.new('1', (self.width + 2, self.height + 2))

                    # Set the pixel value to 1 (default is 0), accounting for padding
                    self.sub_masks[pixel_str].putpixel((x + 1, y + 1), 1)
        self.count_ann = len(self.sub_masks)

    def create_sub_mask_annotation(self, sub_mask,category_id, annotation_id, is_crowd):
        # Find contours (boundary lines) around each sub-mask
        # Note: there could be multiple contours if the object
        # is partially occluded. (E.g. an elephant behind a tree)
        contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

        segmentations = []
        polygons = []
        # plt.figure()
        for contour in contours:
            # Flip from (row, col) representation to (x, y)
            # and subtract the padding pixel
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)

            # Make a polygon and simplify it
            poly = Polygon(contour)
            # # poly = poly.simplify(1.0, preserve_topology=False)
            # # coord = poly.__geo_interface__['coordinates']
            # plt.suptitle('img_id:{0}\t, cat_id:{1}'.format(image_id, primitives[category_id]))
            # plt.subplot(2, 2, 3)
            # xs = np.array(poly.exterior.coords)[:, 0]
            # ys = np.array(poly.exterior.coords)[:, 1]
            # plt.title('before simpility')
            # plt.axis('equal')
            # # plt.xlim(0, 640)
            # # plt.ylim(0, 480)
            # plt.plot(xs, ys)
            # plt.subplot(2, 2, 2)
            # plt.plot(sub_mask)
            # plt.imshow(sub_mask, origin='lower')
            # # plt.plotfile('/home/hamed/Desktop/tutorials/datasets/cig_butts/train/images/color_image_000000.png')
            # plt.subplot(2, 2, 1)
            # print(sorted(files)[image_id])
            # original_image = Image.open('/home/hamed/Desktop/tutorials/datasets/cig_butts/train/images/' + sorted(files)[image_id])
            # plt.imshow(original_image, origin='lower')
            # plt.subplot(2, 2, 4)
            # poly = poly.simplify(0.5, preserve_topology=False)
            # xs = np.array(poly.exterior.coords)[:, 0]
            # ys = np.array(poly.exterior.coords)[:, 1]
            # plt.title('after simpility')
            # plt.axis('equal')
            # plt.plot(xs, ys)

            polygons.append(poly)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)
        # plt.show()

        # Combine the polygons to calculate the bounding box and area
        multi_poly = MultiPolygon(polygons)
        try:
            x, y, max_x, max_y = multi_poly.bounds
            width = max_x - x
            height = max_y - y
            bbox = (x, y, width, height)
            area = multi_poly.area
        except:
            width = 0
            height = 0
            bbox = (0, 0, 0, 0)
            area = 0

        annotation = {
            'segmentation': segmentations,
            'iscrowd': is_crowd,
            'image_id': self.id,
            'category_id': category_id,
            'id': annotation_id,
            'bbox': bbox,
            'area': area
        }
        return annotation
    def imsave(self,dest_root, stage):
        rgb_path = os.path.join(dest_root, stage, 'rgb', self.name)
        depth_path = os.path.join(dest_root, stage,'depth', 'depth_image_' + self.name.split('.png')[0].split('_')[2] + '.png')
        self.rgb.save(rgb_path)
        self.depth.save(depth_path)

    def domain_adapt(self):
        pass

