import numpy as np
import cv2
import os
import glob

import argparse

def generate_mask_from_labels(labels_path, image_path, output_path, target_labels=None, iszero=False):
    # 读取标签文件
    labels = np.load(labels_path)  # 读取标签文件，假设它是一个 .npy 格式的类别 ID 数组
    print(labels)
    # 读取图像
    image = cv2.imread(image_path)
    # 如果没有指定目标标签，默认为全部类别
    if target_labels is None:
        target_labels = np.unique(labels)
    
    # 创建一个掩码，初始化为全黑（全零）
    # mask = np.ones_like(image) * 255
    if iszero:
        mask = np.zeros_like(image) * 255
    else:
        mask = np.ones_like(image) * 255
    # 遍历目标标签
    for label in target_labels:
        # 为当前标签创建掩码
        label_mask = (labels == label)  # 为每个标签创建一个布尔数组
        
        mask[label_mask] = image[label_mask]
    
    # 生成并保存掩码图像
    cv2.imwrite(output_path, mask)

def generate_mask_black_white(labels_path, image_path, output_path, target_labels=None, iszero=True):
    # 读取标签文件
    labels = np.load(labels_path)  # 读取标签文件，假设它是一个 .npy 格式的类别 ID 数组
    print(labels)
    # 读取图像
    image = cv2.imread(image_path)
    # 如果没有指定目标标签，默认为全部类别
    if target_labels is None:
        target_labels = np.unique(labels)
    
    # 创建一个掩码，初始化为全黑（全零）
    # mask = np.ones_like(image) * 255
    
    white_mask = np.ones_like(image) * 255
    if iszero:
        mask = np.zeros_like(image) * 255
    else:
        mask = np.ones_like(image) * 255
    # 遍历目标标签
    for label in target_labels:
        # 为当前标签创建掩码
        label_mask = (labels == label)  # 为每个标签创建一个布尔数组
        
        mask[label_mask] = white_mask[label_mask]
    
    # 生成并保存掩码图像
    cv2.imwrite(output_path, mask)


# # 示例使用
# # labels_path = '/mnt/data3/ysq/smpl_uv/uv_mask/sapiens/test-vrc/01/seg_mask/yonghu_seg.npy'  # 替换为你的标签文件路径
# # image_path = '/mnt/data3/ysq/smpl_uv/uv_mask/sapiens/test-vrc/01/images/yonghu.jpg'  # 替换为你的图像文件路径

# # labels_path = '/mnt/data3/ysq/smpl_uv/uv_mask/sapiens/test-vrc/00/seg_mask/test_seg.npy'  # 替换为你的标签文件路径
# # image_path = '/mnt/data3/ysq/smpl_uv/uv_mask/sapiens/test-vrc/00/images/test.png'  # 替换为你的图像文件路径

# labels_path = '/mnt/data3/ysq/smpl_uv/uv_mask/sapiens/test-vrc/black/seg_mask/black_yonghu_seg.npy'  # 替换为你的标签文件路径
# image_path = '/mnt/data3/ysq/smpl_uv/uv_mask/sapiens/test-vrc/black/images/black_yonghu.png'  # 替换为你的图像文件路径

# # 指定要提取的标签（如果没有提供，则提取所有标签）


# # generate_mask_from_labels(labels_path, image_path, "skin_yonghu.png", skin_labels)
# # generate_mask_from_labels(labels_path, image_path, "skin_yonghu_black.png", skin_labels)
# # generate_mask_from_labels(labels_path, image_path, "skin_none.png", none_skin_labels)
# # generate_mask_from_labels(labels_path, image_path, "skin.png", skin_labels)


#intput_base_dirs = '/mnt/data3/ysq/smpl_uv/uv_mask/sapiens/vrc_data/0414_head_data/input'
#output_base_dir = '/mnt/data3/ysq/smpl_uv/uv_mask/sapiens/vrc_data/0414_head_data/output'
# for no in os.listdir(intput_base_dirs):
#     output_path = os.path.join(output_base_dir, no)
#     os.makedirs(output_path, exist_ok=True)
#     intput_paths = os.path.join(intput_base_dirs, no)
#     for img_name in os.listdir(os.path.join(intput_paths, "images")):
#         seg_path = os.path.join(intput_paths, "seg_mask",f"{os.path.splitext(img_name)[0]}_seg.npy")
#         img_path = os.path.join(intput_paths, "images",img_name)
#         print(img_path)
#         generate_mask_from_labels(seg_path, img_path, os.path.join(output_path, f"{os.path.splitext(img_name)[0]}_skin.{os.path.splitext(img_name)[1]}"), skin_labels)
#         generate_mask_from_labels(seg_path, img_path, os.path.join(output_path, f"{os.path.splitext(img_name)[0]}_noneskin.{os.path.splitext(img_name)[1]}"), none_skin_labels)
#         generate_mask_black_white(seg_path, img_path, os.path.join(output_path, f"{os.path.splitext(img_name)[0]}_skin_mask.{os.path.splitext(img_name)[1]}"), skin_labels)
#         generate_mask_black_white(seg_path, img_path, os.path.join(output_path, f"{os.path.splitext(img_name)[0]}_obj_mask.{os.path.splitext(img_name)[1]}"), obj_labels)

parser = argparse.ArgumentParser(description='Generate masks from segmentation results')
parser.add_argument('--labels_path', type=str, help='Path to _seg.npy file')
parser.add_argument('--image_path', type=str, help='Path to original image')
parser.add_argument('--output_path', type=str, help='Base output directory path')
args = parser.parse_args()

skin_labels = [2, 4, 5, 6, 7, 13, 14, 15, 16, 21, 19, 20, 10, 11]  

none_skin_labels = [0, 1, 3, 8, 9, 12, 17, 18, 22, 23, 24, 25, 26, 27]  

obj_labels = [i for i in range(1, 28)]

if __name__ == "__main__":
    # 从参数获取输入信息
    base_name = os.path.basename(args.image_path)
    file_name = os.path.splitext(base_name)[0]
    ext = os.path.splitext(base_name)[1]
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 生成4种掩码
    generate_mask_from_labels(
        args.labels_path, 
        args.image_path,
        os.path.join(args.output_path, f"{file_name}_skin{ext}"),
        skin_labels
    )
    
    generate_mask_from_labels(
        args.labels_path,
        args.image_path,
        os.path.join(args.output_path, f"{file_name}_noneskin{ext}"),
        none_skin_labels
    )
    
    generate_mask_black_white(
        args.labels_path,
        args.image_path,
        os.path.join(args.output_path, f"{file_name}_skin_mask{ext}"),
        skin_labels
    )