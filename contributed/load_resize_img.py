import os
from scipy import misc


def load_images_resize(src_folder, dst_folder, scale = 8):
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)

    for filename in os.listdir(src_folder):
        img = misc.imread(os.path.join(src_folder, filename))
        if img is not None:
            resize_img = misc.imresize(img, (720, 1280), interp='bilinear')
            filename = os.path.splitext(os.path.split(filename)[1])[0]
            output_filename_n = os.path.join(dst_folder, filename + '.jpg')
            misc.imsave(output_filename_n, resize_img)


load_images_resize('/home/lyh/Datasets/Resize_test/src2','/home/lyh/Datasets/Resize_test/out5')