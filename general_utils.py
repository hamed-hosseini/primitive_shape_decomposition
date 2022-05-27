import numpy as np
color_space = np.asarray([[78.0, 121.0, 167.0], # blue
                          [89.0, 161.0, 79.0], # green
                          [156, 117, 95], # brown
                          [242, 142, 43], # orange
                          [237.0, 201.0, 72.0], # yellow
                          # [186, 176, 172], # gray (bef)
                          [191, 239, 69], # lime
                          # [255.0, 87.0, 89.0], # red (bef)
                          [170, 255, 195], # mint
                          [176, 122, 161], # purple
                          [118, 183, 178], # cyan
                          [255, 157, 167]])/255.0 #pink

id_category = {1:'cuboid', 2:'sphere', 3:'semisphere', 4:'cylinder', 5:'ring',
                6:'stick', 7:'box', 8:'cone', 9:'truncatedcone'}

category_id = {'cuboid':1, 'sphere':2, 'semisphere':3, 'cylinder':4, 'ring':5,
               'stick':6, 'box':7, 'cone':8, 'truncatedcone':9}
