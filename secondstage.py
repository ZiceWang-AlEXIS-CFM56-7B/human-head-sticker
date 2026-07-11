# 用鼻子对齐 + 保持脖子长度
# 使用鼻子、左右眼的平均值
# 用于逐步测试
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
import argparse
import torchvision
import sys
import logging
import warnings
import time
import pdb
import tempfile
import config
warnings.filterwarnings("ignore")
from tqdm import tqdm 
from inference import validation
from network import get_model
from mmpose.apis import MMPoseInferencer
from PIL import Image
from mymodels import SegFaceCeleb
from typing import Optional, Tuple
import math
import matplotlib.pyplot as plt



def visualize_face_centers(image, canvas,
                           nose_center, left_eye_center, right_eye_center,
                           nose_center_global, left_eye_center_global, right_eye_center_global,
                           save_dir="./vis_result"):
    """
    可视化截图与原图中三个人脸关键点（鼻子、左眼、右眼）的坐标，并保存结果。

    参数:
    - image: 原始图像 (H×W×C)
    - canvas: 截取的头部图像 (H×W×C)
    - nose_center, left_eye_center, right_eye_center: 截图中局部坐标
    - nose_center_global, left_eye_center_global, right_eye_center_global: 原图中的全局坐标
    - save_dir: 保存结果图的目录
    """

    os.makedirs(save_dir, exist_ok=True)

    # --- 1. 截图上绘制局部关键点 ---
    canvas_draw = canvas.copy()
    for pt, label in zip([nose_center, left_eye_center, right_eye_center],
                         ['nose', 'left_eye', 'right_eye']):
        pt = tuple(int(x) for x in pt)
        cv2.circle(canvas_draw, pt, 4, (0, 0, 255), -1)
        # cv2.putText(canvas_draw, label, (pt[0] + 5, pt[1] - 5),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 显示截图图像
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(canvas_draw, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Head with Face Keypoints")
    plt.axis('off')
    plt.show()

    # 保存截图图
    cv2.imwrite(os.path.join(save_dir, "cropped_with_points.jpg"), canvas_draw)

    # --- 2. 原图上绘制全局关键点 ---
    image_draw = image.copy()
    for pt, label in zip([nose_center_global, left_eye_center_global, right_eye_center_global],
                         ['nose', 'left_eye', 'right_eye']):
        pt = tuple(int(x) for x in pt)
        cv2.circle(image_draw, pt, 4, (0, 0, 255), -1)
        # cv2.putText(image_draw, label, (pt[0] + 5, pt[1] - 5),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 显示原图图像
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB))
    plt.title("Original Image with Global Face Keypoints")
    plt.axis('off')
    plt.show()

    # 保存原图标注图
    cv2.imwrite(os.path.join(save_dir, "original_with_face_points.jpg"), image_draw)

def cut_head(image, image_path, output_path=None, inferencer_human=None, inferencer_face=None):
    """
    以双眼中点为中心，以双肩 x 坐标之差为边长，截取一个正方形头部图像，
    超出原图部分用白色填充。

    参数:
    - image: 已读入的 numpy 图像 (H×W×C) 或灰度图 (H×W)
    - image_path: 原图路径，用于 MMPoseInferencer 推理
    - output_path: 可选，保存裁剪结果的路径
    - inferencer_human: 已初始化的 MMPoseInferencer 对象 human
    - inferencer_face: 已初始化的 MMPoseInferencer 对象 face

    返回:
    - canvas: 裁剪并填充后的正方形图像 (numpy array)
    - eye_mid: 双眼中点坐标 (x, y)
    - shoulder_width: 双肩 x 坐标之差 (float)
    - e2s_distance: 双眼中点到双肩中点的欧氏距离 (float)
    - n2s_distance: 鼻子到双肩中点的欧氏距离 (float)
    - shoulder_mid: 双肩中点坐标 (x, y)
    - nose: 人脸鼻子坐标 (x, y)
    """
    if inferencer_human is None:
        inferencer_human = MMPoseInferencer(pose2d='human')
    
    if inferencer_face is None:
        inferencer_face = MMPoseInferencer(pose2d='face')

    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    # 推理关键点
    result_generator = inferencer_human(image_path, skeleton_style='mmpose')
    result = next(result_generator)
    preds = result['predictions'][0][0].get('keypoints', [])
    if not preds:
        raise ValueError("未检测到人体关键点")

    keypoints = preds  # list of [x, y, score]
    # COCO keypoint idx: 0=nose, 1=left_eye, 2=right_eye, 5=left_shoulder, 6=right_shoulder
    NOSE, LEYE, REYE, LSHOULDER, RSHOULDER = 0, 1, 2, 5, 6

    nose           = np.array(keypoints[NOSE][:2], dtype=float)
    left_eye       = np.array(keypoints[LEYE][:2], dtype=float)
    right_eye      = np.array(keypoints[REYE][:2], dtype=float)
    left_shoulder  = np.array(keypoints[LSHOULDER][:2], dtype=float)
    right_shoulder = np.array(keypoints[RSHOULDER][:2], dtype=float)

    # 检查关键点有效性
    for name, kp in zip(
        ['nose','left_eye','right_eye','left_shoulder','right_shoulder'],
        [nose, left_eye, right_eye, left_shoulder, right_shoulder]
    ):
        if np.any(np.isnan(kp)):
            raise ValueError(f"{name}关键点无效")

    # 计算双眼中点和肩宽
    eye_mid = (left_eye + right_eye) / 2.0
    shoulder_width = float(abs(right_shoulder[0] - left_shoulder[0]))

    # 计算双肩中点
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0

    # 计算距离
    e2s_distance = float(np.linalg.norm(eye_mid - shoulder_mid))
    n2s_distance = float(np.linalg.norm(nose    - shoulder_mid))

    # 确定裁剪正方形的边长和左上角坐标
    side_len = int(round(shoulder_width))
    half = side_len // 2
    cx, cy = int(round(eye_mid[0])), int(round(eye_mid[1]))
    left, top = cx - half, cy - half
    right, bottom = left + side_len, top + side_len

    # 准备白色背景画布
    h, w = image.shape[:2]
    if image.ndim == 3:
        C = image.shape[2]
        canvas = np.full((side_len, side_len, C), 255, dtype=image.dtype)
    else:
        canvas = np.full((side_len, side_len), 255, dtype=image.dtype)

    # 计算在原图和画布上的实际拷贝区域
    src_x1, src_y1 = max(left, 0), max(top, 0)
    src_x2, src_y2 = min(right, w), min(bottom, h)
    dst_x1 = max(0, -left)
    dst_y1 = max(0, -top)
    h_crop = src_y2 - src_y1
    w_crop = src_x2 - src_x1

    if h_crop > 0 and w_crop > 0:
        canvas[dst_y1:dst_y1+h_crop, dst_x1:dst_x1+w_crop] = \
            image[src_y1:src_y1+h_crop, src_x1:src_x1+w_crop]
    
    # 重新识别人脸关键点
    face_vis_dir = os.path.splitext(output_path)[0]
    os.makedirs(face_vis_dir, exist_ok = True)
    
    resultFace_generator = inferencer_face(canvas, skeleton_style='mmpose', vis_out_dir = face_vis_dir)
    result_face = next(resultFace_generator)
    keypoints_face = result_face['predictions'][0][0]['keypoints'] 
    # print(keypoints_face)

    print(len(keypoints_face))
    # 定义关键点索引
    mmpose106_leye_indices = list(range(66, 75))    # 左眼
    mmpose106_reye_indices = list(range(75, 84))    # 右眼
    mmpose106_nose_indices = list(range(52, 66))    # 鼻子

    # 提取关键点坐标
    face_noses = np.array([keypoints_face[i] for i in mmpose106_nose_indices], dtype=float)
    face_left_eyes = np.array([keypoints_face[i] for i in mmpose106_leye_indices], dtype=float)
    face_right_eyes = np.array([keypoints_face[i] for i in mmpose106_reye_indices], dtype=float)

    # 计算平均位置
    nose_center = np.mean(face_noses, axis=0)
    left_eye_center = np.mean(face_left_eyes, axis=0)
    right_eye_center = np.mean(face_right_eyes, axis=0)
    
    nose_center_global      = nose_center + np.array([left, top])
    left_eye_center_global  = left_eye_center + np.array([left, top])
    right_eye_center_global = right_eye_center + np.array([left, top])
    
    # 打印结果
    print("Global Nose center (x, y):", nose_center_global)
    print("Global Left eye center (x, y):", left_eye_center_global)
    print("Global Right eye center (x, y):", right_eye_center_global)
    
    # 可视化纠正后的结果 截图的 全图的
    visualize_face_centers(
        image=image,
        canvas=canvas,
        nose_center=nose_center,
        left_eye_center=left_eye_center,
        right_eye_center=right_eye_center,
        nose_center_global=nose_center_global,           # 全局鼻子
        left_eye_center_global=left_eye_center_global,   # 全局左眼
        right_eye_center_global=right_eye_center_global, # 全局右眼
        save_dir=face_vis_dir
    )
    
    # 可选保存
    if output_path:
        cv2.imwrite(output_path, canvas)
        
    # 再次计算双眼中点
    eye_mid_new = (left_eye_center_global + right_eye_center_global) / 2.0
    
    # 再次计算距离
    e2s_distance_new = float(np.linalg.norm(eye_mid_new - shoulder_mid))
    n2s_distance_new = float(np.linalg.norm(nose_center_global - shoulder_mid))
    
    return canvas, \
           (eye_mid_new[0], eye_mid_new[1]), \
           shoulder_width, \
           e2s_distance_new, \
           n2s_distance_new, \
           (shoulder_mid[0], shoulder_mid[1]), \
           (nose_center_global[0], nose_center_global[1]), \
           (eye_mid[0], eye_mid[1]) # 这是在第 5 步中，模特白头图需要对齐到的点



def gain_nose(eye_mid: tuple, shoulder_width: float, nose_global: tuple) -> tuple:
    """
    计算鼻子在裁剪后 canvas 上的坐标。

    参数:
    - eye_mid: 双眼中点坐标 (x, y)，来自 cut_head 返回值
    - shoulder_width: 裁剪正方形的边长（float），来自 cut_head 返回值
    - nose_global: 原图中鼻子的全局坐标 (x, y)，来自 cut_head 返回值

    返回:
    - nose_canvas: 鼻子在 canvas（裁剪图）上的坐标 (x, y)
    """
    side_len = int(round(shoulder_width))
    half = side_len // 2

    # 原图上裁剪区域左上角坐标
    cx = int(round(eye_mid[0]))
    cy = int(round(eye_mid[1]))
    left = cx - half
    top = cy - half

    # 全局坐标转裁剪图坐标
    nose_x, nose_y = nose_global
    x_in_canvas = nose_x - left
    y_in_canvas = nose_y - top

    return (x_in_canvas, y_in_canvas)



def matting_head(head_square: np.ndarray,
                 model: nn.Module,
                 device: str = "cpu",
                 output_path: Optional[str] = None) -> np.ndarray:
    """
    对 cut_head 输出的正方形头部图像进行抠图。

    步骤：
    1. 记录输入图像的原始尺寸 orig_h, orig_w
    2. 由于输入已为正方形，可直接 resize 到 512×512
    3. 用 SegFace 模型推理，得到语义分割 mask，并后处理
    4. 将带 alpha 通道的 RGBA 图像从 512×512 resize 回 orig_h×orig_w
    5. 可选保存到 output_path

    参数：
    - head_square: cut_head 输出的正方形 BGR 图像 (H×H×3)
    - model: 已加载到 device 上的 SegFaceCeleb 模型
    - device: 运行设备 ("cpu" 或 "cuda")
    - output_path: 若指定，则将最终 RGBA 图像写入该路径

    返回：
    - final_rgba: 与 head_square 同尺寸的 RGBA 图像 (H×H×4)
    """
    # 1. 原始尺寸
    orig_h, orig_w = head_square.shape[:2]

    # 2. 直接缩放到 512×512
    head_512 = cv2.resize(head_square, (512, 512), interpolation=cv2.INTER_AREA)

    # 3. 模型推理生成 mask
    def preprocess(img_bgr: np.ndarray) -> torch.Tensor:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb.transpose(2,0,1)).float() / 255.0
        return tensor.unsqueeze(0).to(device)

    model.to(device).eval()
    with torch.no_grad():
        logits = model(preprocess(head_512))   # [1, C, 512, 512]
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # [512,512]

    # KEEP_CLASSES 同 gain_matting 中定义的那些前景类别
    KEEP_CLASSES = [2,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    mask = np.isin(pred, KEEP_CLASSES).astype(np.uint8) * 255

    # 后处理 mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    # 合成 RGBA
    b, g, r = cv2.split(head_512)
    rgba_512 = cv2.merge([b, g, r, mask])

    # 4. 缩回原始正方形尺寸
    final_rgba = cv2.resize(rgba_512, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # 5. 可选保存
    if output_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # cv2.imwrite 会自动根据扩展名保存带 alpha 通道的 PNG
        cv2.imwrite(output_path, final_rgba)

    return final_rgba


def white_head(
    square_img: np.ndarray,
    model: torch.nn.Module,
    device: str = "cpu",
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    对 cut_head 输出的正方形人头图像进行白头处理：
    1. 将正方形图像缩放至 512×512
    2. 使用 SegFace 模型预测分割
    3. 将预测结果中属于 KEEP_CLASSES 的区域填充为白色，其它像素保持原色
    4. 将结果缩放回原始正方形尺寸并返回

    参数：
    - square_img: cut_head 输出的正方形 BGR 图像 (H×H×3)
    - model: 已加载并置于 eval 模式的 SegFaceCeleb 模型
    - device: "cpu" 或 "cuda"
    - output_path: 若指定，则保存最终结果到此路径

    返回：
    - final_img: 与输入同尺寸的 BGR 图像（已应用白头效果）
    """
    # 记录原始正方形尺寸
    orig_h, orig_w = square_img.shape[:2]
    assert orig_h == orig_w, "输入必须是正方形图像"

    # 1. 缩放至 512×512
    img_512 = cv2.resize(square_img, (512, 512), interpolation=cv2.INTER_AREA)

    # 2. 分割预测
    model.to(device)
    model.eval()
    # 预处理
    img_rgb = cv2.cvtColor(img_512, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb.transpose(2,0,1)).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)                    # [1, num_classes, 512, 512]
        pred = logits.argmax(dim=1).squeeze(0)    # [512, 512]，每点类别索引

    # 3. 填充 KEEP_CLASSES 区域为白色
    KEEP_CLASSES = [2,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    mask = np.isin(pred.cpu().numpy(), KEEP_CLASSES)  # bool mask
    # 将 mask 区域设为白
    img_512[mask] = [255, 255, 255]

    # 4. 缩放回原始尺寸
    final_square = cv2.resize(img_512, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # 可选保存
    if output_path:
        cv2.imwrite(output_path, final_square)

    return final_square



def align_model(
    headimg: np.ndarray,
    modelimg: np.ndarray,
    modelpoint: Tuple[float, float],
    userpoint: Tuple[float, float],
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    将头部图 headimg 放置到模特图 modelimg 上，使得 headimg 上的 userpoint 
    对齐到 modelimg 上的 modelpoint。

    参数：
    - headimg: 头部图像 (Hh×W_h×3 或 Hh×W_h×4)，可以带或不带 alpha 通道
    - modelimg: 模特图像 (Hm×Wm×3)
    - modelpoint: 在 modelimg 上需要对齐的目标点 (x, y)
    - userpoint: 在 headimg 上需要对齐的点 (x, y)
    - output_path: 可选，将合成结果保存到该路径

    返回：
    - composite: 合成后的图像，尺寸与 modelimg 一致
    """
    # 创建模特图像的副本
    composite = modelimg.copy()
    
    # 计算头部图像应该放置的位置
    # 计算偏移量：modelpoint - userpoint
    offset_x = int(modelpoint[0] - userpoint[0])
    offset_y = int(modelpoint[1] - userpoint[1])
    
    # 确定头部图像在模特图像上的位置
    hh, wh = headimg.shape[:2]
    hm, wm = modelimg.shape[:2]
    
    # 计算放置区域
    x1 = max(offset_x, 0)
    y1 = max(offset_y, 0)
    x2 = min(offset_x + wh, wm)
    y2 = min(offset_y + hh, hm)
    
    # 计算头部图像的有效区域
    head_x1 = max(-offset_x, 0)
    head_y1 = max(-offset_y, 0)
    head_x2 = min(wm - offset_x, wh)
    head_y2 = min(hm - offset_y, hh)
    
    # 检查是否有重叠区域
    if x1 >= x2 or y1 >= y2 or head_x1 >= head_x2 or head_y1 >= head_y2:
        return composite
    
    # 处理头部图像（带或不带alpha通道）
    if headimg.shape[2] == 4:  # 带有alpha通道
        # 提取alpha通道并归一化
        alpha = headimg[head_y1:head_y2, head_x1:head_x2, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        
        # 提取RGB通道
        head_rgb = headimg[head_y1:head_y2, head_x1:head_x2, :3]
        
        # 进行alpha混合
        composite[y1:y2, x1:x2] = (head_rgb * alpha + 
                                  composite[y1:y2, x1:x2] * (1 - alpha)).astype(np.uint8)
    else:  # 不带alpha通道
        composite[y1:y2, x1:x2] = headimg[head_y1:head_y2, head_x1:head_x2]
    
    # 可选保存
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, composite)
    
    return composite



if __name__ == "__main__":
    root_dir = "/root/autodl-tmp/assets"
    
    # 关键点检测
    inferencer_human = MMPoseInferencer(pose2d='human')
    inferencer_face = MMPoseInferencer(pose2d='face')
    # 分割模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_model = SegFaceCeleb(512, "convnext_base").to(device)
    ckpt = torch.load("weights/convnext_celeba_512/model_299.pt", map_location=device)
    seg_model.load_state_dict(ckpt["state_dict_backbone"])
    seg_model.eval()
    
    # —— 全局计时开始 ——
    total_start = time.time()
    print(root_dir)
    for sub in sorted(os.listdir(root_dir), key=lambda x: (0, int(x)) if x.isdigit() else (1, x)):
        subdir = os.path.join(root_dir, sub)
        
        # print(f"=== Processing folder {sub} ===")
        def print_boxed(text):
            border = "=" * (len(text) + 6)
            print(f"\n{border}")
            print(f"== {text} ==")
            print(f"{border}\n")
        print_boxed(f"Processing folder: {sub}")
        
        # —— 单文件夹计时开始 ——
        folder_start = time.time()
        
        # —— 第 1 步: cut_head 函数处理 ct/model_ct.jpg ——
        print(f"[{sub}] ===正在处理第 1 步 cut_head -> ct/model_ct.jpg... ...===")
        model_ct_path = os.path.join(subdir, "ct", "model_ct.jpg")
        # 读取原图
        model_ct_image = cv2.imread(model_ct_path)
        if model_ct_image is None:
            raise FileNotFoundError(f"无法读取图像: {model_ct_path}")

        # 准备 head 保存目录
        model_head_dir = os.path.join(subdir, "head")
        os.makedirs(model_head_dir, exist_ok=True)
        # 裁剪后保存的路径
        model_head_path = os.path.join(model_head_dir, "model_head.jpg")
        
        try:
            # 调用 cut_head，内部会自动调用 visualize_face_centers 并把 cropped_with_points.jpg 与 original_with_face_points.jpg 保存在 head_dir/model 下
            canvas, eye_mid_model, shoulder_width_model, e2s_distance_model, n2s_distance_model, shoulder_mid_model, nose_model, white_head_alignpoint = \
                cut_head(
                    image=model_ct_image,
                    image_path=model_ct_path,
                    output_path=model_head_path,
                    inferencer_human=inferencer_human,
                    inferencer_face=inferencer_face
                )
            
            # 计算模特鼻子在 canvas 上的坐标
            nose_canvas_model = gain_nose(eye_mid_model, shoulder_width_model, nose_model)
            print(f"模特鼻子在大头照上的坐标={nose_canvas_model}")

            # 打印六个返回值
            print(f"[{sub}] 模特： "
                  f"双眼中点={eye_mid_model}, 肩宽={shoulder_width_model}, \n"
                  f"双眼中点到双肩中点的距离={e2s_distance_model}, 鼻子到双肩中点的距离={n2s_distance_model}, \n"
                  f"双肩中点={shoulder_mid_model}, 鼻子={nose_model} \n")
            print(f"[{sub}] 第 1 步 cut_head -> ct/model_ct.jpg 完成")
            
        except Exception as e:
            print(f"[{sub}] 第 1 步 cut_head -> ct/model_ct.jpg 失败: {e}")
            continue
            
        # —— 第 2 步: cut_head 函数处理 user.jpg ——
        print(f"[{sub}] ===正在处理第 2 步 cut_head -> user.jpg... ...===")
        user_path = os.path.join(subdir, "user.jpg")
        # 读取原图
        user_image = cv2.imread(user_path)
        if user_image is None:
            raise FileNotFoundError(f"无法读取图像: {user_path}")

        # 准备 head 保存目录
        user_head_dir = os.path.join(subdir, "head")
        os.makedirs(user_head_dir, exist_ok=True)
        # 裁剪后保存的路径
        user_head_path = os.path.join(user_head_dir, "user_head.jpg")
        
        try:
            # 调用 cut_head，内部会自动调用 visualize_face_centers 并把 cropped_with_points.jpg 与 original_with_face_points.jpg 保存在 head_dir/model 下
            canvas, eye_mid_user, shoulder_width_user, e2s_distance_user, n2s_distance_user, shoulder_mid_user, nose_user, _ = \
                cut_head(
                    image=user_image,
                    image_path=user_path,
                    output_path=user_head_path,
                    inferencer_human=inferencer_human,
                    inferencer_face=inferencer_face
                )
            
            # 计算用户鼻子在 canvas 上的坐标
            nose_canvas_user = gain_nose(eye_mid_user, shoulder_width_user, nose_user)
            print(f"用户鼻子在大头照上的坐标={nose_canvas_user}")

            # 打印六个返回值
            print(f"[{sub}] 用户： "
                  f"双眼中点={eye_mid_user}, 肩宽={shoulder_width_user}, \n"
                  f"双眼中点到双肩中点的距离={e2s_distance_user}, 鼻子到双肩中点的距离={n2s_distance_user}, \n"
                  f"双肩中点={shoulder_mid_user}, 鼻子={nose_user} \n")
            print(f"[{sub}] 第 2 步 cut_head -> user.jpg 完成")
        
        except Exception as e:
            print(f"[{sub}] 第 2 步 cut_head -> user.jpg 失败: {e}")
            continue
                        
        # —— 第 3 步: white_head 函数处理 head/model_head.jpg ——
        print(f"[{sub}] ===正在处理第 3 步 white_head -> head/model_head.jpg... ...===")
        inter_dir = os.path.join(subdir, "inter")
        os.makedirs(inter_dir, exist_ok=True)
        model_head_square = cv2.imread(model_head_path)
        white_jpg = os.path.join(inter_dir, "model_head_white.jpg")
        try:
            white_head(
                square_img=model_head_square,
                model=seg_model,
                device=device,
                output_path=white_jpg
            )
            print(f"[{sub}] 第 3 步 white_head -> head/model_head.jpg 完成")
        except Exception as e:
            print(f"[{sub}] 第 3 步 white_head -> head/model_head.jpg 失败: {e}")
            continue
            
        # —— 第 4 步: matting_head 函数处理 head/user_head.jpg ——
        print(f"[{sub}] ===正在处理第 4 步 matting_head -> head/user_head.jpg... ...===")
        user_head_square = cv2.imread(user_head_path)
        matting_png = os.path.join(inter_dir, "user_head_matting.png")
        try:
            matting_head(
                head_square=user_head_square,
                model=seg_model,
                device=device,
                output_path=matting_png
            )
            print(f"[{sub}] 第 4 步 matting_head -> head/user_head.jpg 完成")
        except Exception as e:
            print(f"[{sub}] 第 4 步 matting_head -> head/user_head.jpg 失败: {e}")
            continue
            
        # —— 第 5 步: align_model 函数得到无头模特 ——
        print(f"[{sub}] ===正在处理第 5 步 align_model... ...===")
        wohead_dir = os.path.join(subdir, "wohead")
        os.makedirs(wohead_dir, exist_ok=True)
        wohead_jpg = os.path.join(wohead_dir, "model_wohead.jpg")
        try:
            head_white_img = cv2.imread(white_jpg, cv2.IMREAD_UNCHANGED)
            model_img = cv2.imread(model_ct_path)
            if head_white_img is None or model_img is None:
                raise RuntimeError("读图失败")
            
            # 以截图的、旧的双眼中点作为 modelpoint
            # 以模特肩宽的一半作为 userpoint
            align_model(
                headimg=head_white_img,
                modelimg=model_img,
                modelpoint=white_head_alignpoint,
                userpoint=(shoulder_width_model/2.0, shoulder_width_model/2.0),
                output_path=wohead_jpg
            )
            print(f"[{sub}] 第 5 步 align_model 完成，输出：{wohead_jpg}")
        except Exception as e:
            print(f"[{sub}] 第 5 步 align_model 失败: {e}")
            continue
            
        # —— Step 6: align_user combine ——
        print(f"[{sub}] ===正在处理第 6 步 align_user... ...===")
        try:
            # 读取无头模型图和用户抠图
            wohead_img = cv2.imread(wohead_jpg)
            user_mat = cv2.imread(matting_png, cv2.IMREAD_UNCHANGED)
            if wohead_img is None or user_mat is None:
                raise RuntimeError("读取图像失败")
            
            ### 得到缩放后的用户透明头 ###
            # 用整数的裁剪边长来保证 scale 精确
            side_len_model = int(round(shoulder_width_model))
            side_len_user  = int(round(shoulder_width_user))
            # resize 时也用 side_len_model
            user_resized = cv2.resize(user_mat,
                                      (side_len_model, side_len_model),
                                      interpolation=cv2.INTER_AREA)
            
            # 保存resize后的用户头像到inter文件夹
            user_resized_path = os.path.join(inter_dir, "user_head_resized.png")
            cv2.imwrite(user_resized_path, user_resized)

            ### 得到缩放后的用户鼻子坐标 ###
            # 精确的缩放比例
            scale = side_len_model / side_len_user
            # 计算缩放后的用户鼻子坐标
            nose_user_resized = (
                nose_canvas_user[0] * scale,
                nose_canvas_user[1] * scale
            )
            print(f"缩放后的用户鼻子坐标={nose_user_resized}")
            
            # 请定义一个点 starpoint
            # starpoint 的 x 坐标就是 nose_user_resized 的 x 坐标
            # starpoint 的 y 坐标 = shoulder_mid_model 的 y 坐标 - n2s_distance_user * scale
            starpoint = (
                nose_model[0],
                shoulder_mid_model[1] - n2s_distance_user * scale
            )
            print(f"按照用户的脖子长度，换算到模特图上的脖子长度={n2s_distance_user * scale}")
            print(f"模特图上将要对齐的starpoint：{starpoint}")

            final_output = os.path.join(subdir, "combined.jpg")
            # 利用 align_model 函数：headimg 是 user_resized，modelimg 是 wohead_img，
            # modelpoint 是 nose_canvas_model，userpoint 是 nose_user_resized
            align_model(
                headimg=user_resized,
                modelimg=wohead_img,
                # modelpoint=nose_model,
                modelpoint=starpoint,
                userpoint=nose_user_resized,
                output_path=final_output
            )

            print(f"[{sub}] 第 6 步 align_user 完成，输出：{final_output}")
        except Exception as e:
            print(f"[{sub}] 第 6 步 align_user 失败: {e}")
            continue

            
        # —— 单文件夹计时结束 ——
        folder_end = time.time()
        print(f"[{sub}] 本文件夹耗时: {folder_end - folder_start:.2f} 秒\n")
        
        
    # —— 全局计时结束 ——
    total_end = time.time()
    print(f"全部文件夹处理完毕，总耗时: {total_end - total_start:.2f} 秒")
    
    print("全部文件夹处理完毕。")
