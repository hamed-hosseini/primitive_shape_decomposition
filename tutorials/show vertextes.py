from matplotlib import pyplot as plt
import matplotlib
from PIL import Image
import json
# import tkinter
# matplotlib.use('TkAgg')
if __name__=="__main__":
    fig, ax = plt.subplots()
    file_name = '/home/hosseini/Desktop/grasp_primitiveShape/tutorials/datasets/primitive_shapes/train/rgb/color_image_000006.png'
    img = Image.open(file_name)
    ax.imshow(img)
    f = open('/home/hosseini/Desktop/grasp_primitiveShape/tutorials/datasets/primitive_shapes/train/coco_annotations.json')
    json_file = json.load(f)
    print(json_file)
    # my_dict = json.loads(json_file)
    print(json_file['annotations'][3]['segmentation'])
    all_vertexes = json_file['annotations'][3]['segmentation']
    for vertexes in all_vertexes:
        x = vertexes[0::2]
        y = vertexes[1::2]
        ax.plot(x, y)
    plt.show()

