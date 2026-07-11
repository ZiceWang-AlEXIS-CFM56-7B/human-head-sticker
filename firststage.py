# 咏航师弟做的、把橘色衣服聚类掉的版本
# ---------- 导入依赖模块 ----------
import os
import sys
import subprocess
import glob
import cv2
import numpy as np
import time
from sklearn.cluster import KMeans

# 颜色迁移模块导入
color_transfer_path = "/root/autodl-tmp/color-transfer0430/code"
sys.path.insert(0, color_transfer_path)  # 修复这里，添加了索引0

try:
    from color_transfer import color_transfer
except ImportError as e:
    print(f"无法导入color_transfer模块，请检查路径: {color_transfer_path}")
    print(f"错误详情: {str(e)}")
    exit(1)


# ---------- 路径配置 ----------
BASE_DIR = "/root/autodl-tmp"  # 根目录

# 关键路径配置（移除了pose_checkpoint和vis_pose_script）
PATH_CONFIG = {
    "seg_checkpoint": os.path.join(BASE_DIR, 
        "sapiens_lite_host/torchscript/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"),
    "vis_seg_script": os.path.join(BASE_DIR, "sapiens/lite/demo/vis_seg.py"),
    "read_npy_script": os.path.join(BASE_DIR, "sapiens/lite/demo/read_npy.py"),
    "input_dir": os.path.join(BASE_DIR, "assets"),  # 输入数据目录
    "output_dir": os.path.join(BASE_DIR, "tmp/seg_pipeline"),  # 中间输出目录
}


def validate_paths():
    """路径验证函数（移除了姿态估计相关路径检查）"""
    required_paths = [
        ("seg_checkpoint", PATH_CONFIG["seg_checkpoint"]),
        ("vis_seg_script", PATH_CONFIG["vis_seg_script"]),
        ("read_npy_script", PATH_CONFIG["read_npy_script"]),
        ("input_dir", PATH_CONFIG["input_dir"])
    ]
    
    missing = []
    for name, path in required_paths:
        if not os.path.exists(path):
            missing.append(f"{name}: {path}")
    
    if missing:
        print("以下关键路径不存在：")
        print("\n".join(missing))
        exit(1)


# ---------- 优化 K-means 皮肤掩码修正函数 ----------
def refine_skin_mask(skin_img_path, skin_mask_path):
    """优化后的K-means皮肤掩码修正"""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("需要安装scikit-learn库: pip install scikit-learn")
        return None

    # 读取数据并验证
    skin_img = cv2.imread(skin_img_path)
    orig_mask = cv2.imread(skin_mask_path, 0)
    if skin_img is None or orig_mask is None:
        print("无法读取图像或掩码文件")
        return None

    # 转换为更适合皮肤分析的YCrCb颜色空间
    ycrcb_img = cv2.cvtColor(skin_img, cv2.COLOR_BGR2YCrCb)
    
    # 提取原始掩码区域像素
    mask_area = (orig_mask > 128)
    pixels = ycrcb_img[mask_area].astype(np.float32)
    
    if len(pixels) < 100:  # 极小样本情况
        return orig_mask

    # 动态确定聚类数量（基于样本量）
    n_clusters = 3 if len(pixels) > 3000 else 2
    
    # 下采样策略优化
    sample_size = min(10000, len(pixels))
    np.random.seed(42)
    samples = pixels[np.random.choice(len(pixels), sample_size, replace=False)]

    # K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(samples)
    
    # 获取所有像素标签
    labels = kmeans.predict(pixels)

    # 计算各簇特征
    cluster_stats = []
    for i in range(n_clusters):
        cluster_pixels = pixels[labels == i]
        y_channel = cluster_pixels[:, 0]
        
        # 统计特征
        stats = {
            "size": len(cluster_pixels),
            "mean_Y": np.mean(y_channel),
            "std_Y": np.std(y_channel),
            "mean_Cr": np.mean(cluster_pixels[:, 1]),
            "mean_Cb": np.mean(cluster_pixels[:, 2])
        }
        cluster_stats.append(stats)

    # ===== 关键参数配置 =====
    TUNE_PARAMS = {
        "color_range": {
            "Cr": (130, 170),  # 扩展Cr范围
            "Cb": (85, 130)    # 扩展Cb范围
        },
        "brightness_range": (30, 220),    # 扩展亮度范围
        "min_cluster_ratio": 0.1,        # 降低最小簇比例
        "morph_kernel_size": 5,           # 减小形态学核尺寸
        "morph_iterations": 2             # 减少形态学操作次数
    }

    # 筛选策略优化（使用新参数）
    valid_clusters = []
    for idx, stats in enumerate(cluster_stats):
        is_valid = True
        
        # 颜色范围检查
        cr_cond = TUNE_PARAMS["color_range"]["Cr"][0] < stats["mean_Cr"] < TUNE_PARAMS["color_range"]["Cr"][1]
        cb_cond = TUNE_PARAMS["color_range"]["Cb"][0] < stats["mean_Cb"] < TUNE_PARAMS["color_range"]["Cb"][1]
        if not (cr_cond and cb_cond):
            is_valid = False
            
        # 亮度范围检查
        y_cond = TUNE_PARAMS["brightness_range"][0] < stats["mean_Y"] < TUNE_PARAMS["brightness_range"][1]
        if not y_cond:
            is_valid = False
            
        # 最小簇大小检查
        if stats["size"] < TUNE_PARAMS["min_cluster_ratio"] * len(pixels):
            is_valid = False
            
        if is_valid:
            valid_clusters.append(idx)
            
    # 创建新掩码
    new_mask = np.zeros_like(orig_mask)
    if valid_clusters:
        valid_pixels = np.isin(labels, valid_clusters)
        coords = np.argwhere(mask_area)
        new_mask[coords[valid_pixels, 0], coords[valid_pixels, 1]] = 255
    else:
        new_mask = orig_mask.copy()

    # 改进的形态学处理（使用新参数）
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (TUNE_PARAMS["morph_kernel_size"], TUNE_PARAMS["morph_kernel_size"])
    )
    new_mask = cv2.morphologyEx(
        new_mask, 
        cv2.MORPH_CLOSE, 
        kernel, 
        iterations=TUNE_PARAMS["morph_iterations"]
    )    
    # 去除小区域（面积<500像素）
    contours, _ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            cv2.drawContours(new_mask, [cnt], 0, 0, -1)

    return new_mask


# ---------- 颜色迁移处理函数 ----------
def process_color_transfer(ct_dir):
    """颜色迁移处理函数"""
    try:
        model_dir = os.path.join(ct_dir, "model")
        user_dir = os.path.join(ct_dir, "user")

        if not os.path.exists(model_dir) or not os.path.exists(user_dir):
            raise FileNotFoundError("缺少model或user子目录")

        file_paths = {
            "source_skin": os.path.join(user_dir, "user_skin.*"),
            "target_skin": os.path.join(model_dir, "model_skin.*"),
            "tcloth": os.path.join(model_dir, "model_noneskin.*"),
            "ssmask": os.path.join(user_dir, "user_skin_mask.*"),
            "tsmask": os.path.join(model_dir, "model_skin_mask.*")
        }

        files = {}
        for key, pattern in file_paths.items():
            matches = glob.glob(pattern)
            if not matches:
                raise FileNotFoundError(f"未找到 {key} 文件，路径模式: {pattern}")
            files[key] = matches[0]

        source = cv2.imread(files["source_skin"])
        target = cv2.imread(files["target_skin"])
        tcloth = cv2.imread(files["tcloth"])
        ssmask = cv2.imread(files["ssmask"], 0)
        tsmask = cv2.imread(files["tsmask"], 0)

        ssmask = 255 * (ssmask > 128).astype(np.uint8)
        tsmask = 255 * (tsmask > 128).astype(np.uint8)

        transfer = color_transfer(
            source=source,
            ssmask=ssmask,
            target=target,
            tsmask=tsmask,
            clip=True,
            preserve_paper=False
        )

        result = tcloth.copy()
        result[tsmask > 0] = transfer[tsmask > 0]
        
        output_path = os.path.join(ct_dir, "model_ct.jpg")
        cv2.imwrite(output_path, result)
        print(f"颜色迁移结果已保存至：{output_path}")
        return True
        
    except Exception as e:
        print(f"颜色迁移失败: {str(e)}")
        return False


# ---------- 查找原始图像函数 ----------
def find_original_image(input_dir, base_name):
    """查找原始图片"""
    for ext in ['.jpg', '.png', '.jpeg']:
        candidate = os.path.join(input_dir, f"{base_name}{ext}")
        if os.path.exists(candidate):
            return candidate
    print(f"未找到原始图片: {base_name}")
    return None


# ---------- 处理单个子目录的主流程 ----------
def process_single_directory(sub_dir):
    """对一个数字命名的子目录依次做分割、掩码优化和颜色迁移"""
    start_time = time.time()
    
    input_dir = os.path.join(PATH_CONFIG["input_dir"], sub_dir)
    output_seg_dir = os.path.join(PATH_CONFIG["output_dir"], sub_dir)
    final_output_dir = os.path.join(input_dir, "ct")
    
    os.makedirs(output_seg_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)

    print(f"[{sub_dir}] 开始处理...")
    
    # 执行分割
    try:
        subprocess.run([
            "python", PATH_CONFIG["vis_seg_script"],
            PATH_CONFIG["seg_checkpoint"],
            "--input", input_dir,
            "--output_root", output_seg_dir,
            "--batch_size", "2"
        ], check=True, timeout=300)
    except Exception as e:
        print(f"[{sub_dir}] 分割失败: {str(e)}")
        return False

    # 处理分割结果
    seg_files = glob.glob(os.path.join(output_seg_dir, "*_seg.npy"))
    if not seg_files:
        print(f"[{sub_dir}] 未找到分割结果文件")
        return False

    # 生成皮肤/掩码
    success_count = 0
    for seg_path in seg_files:
        base_name = os.path.basename(seg_path).replace("_seg.npy", "")
        img_path = find_original_image(input_dir, base_name)
        if not img_path:
            continue

        try:
            output_subdir = os.path.join(final_output_dir, base_name)
            os.makedirs(output_subdir, exist_ok=True)
            
            # 执行read_npy_script
            subprocess.run([
                "python", PATH_CONFIG["read_npy_script"],
                "--labels_path", seg_path,
                "--image_path", img_path,
                "--output_path", output_subdir
            ], check=True, timeout=60)

            # 获取原始图片扩展名
            img_ext = os.path.splitext(img_path)[1].lower()
            
            # 构造预期文件名模式
            expected_files = {
                "user_skin": f"{base_name}_skin{img_ext}",
                "user_skin_mask": f"{base_name}_skin_mask{img_ext}"
            }

            # 查找生成的文件
            user_skin_path = os.path.join(output_subdir, expected_files["user_skin"])
            user_mask_path = os.path.join(output_subdir, expected_files["user_skin_mask"])

            # 添加存在性检查
            if not os.path.exists(user_skin_path):
                print(f"皮肤文件未找到: {user_skin_path}")
                continue
            if not os.path.exists(user_mask_path):
                print(f"掩码文件未找到: {user_mask_path}")
                continue

            # 优化掩码
            debug_dir = os.path.join(output_subdir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            # 判断是否为用户数据（检查文件名是否包含'user'）
            is_user_data = 'user' in base_name.lower()
            
            if is_user_data:
                refined_mask = refine_skin_mask(user_skin_path, user_mask_path)
                if refined_mask is not None:
                    cv2.imwrite(user_mask_path, refined_mask)
                    cv2.imwrite(os.path.join(debug_dir, "refined_mask.png"), refined_mask)
            else:
                # 对于非用户数据（如model），直接使用原始掩码，不进行优化
                pass

            success_count += 1
            
        except Exception as e:
            print(f"[{sub_dir}] {base_name} 处理失败: {str(e)}")

    # 执行颜色迁移
    if success_count >= 2:
        color_result = process_color_transfer(final_output_dir)
        elapsed_time = time.time() - start_time
        print(f"[{sub_dir}] 处理完成，耗时: {elapsed_time:.2f}秒")
        return color_result
    
    elapsed_time = time.time() - start_time
    print(f"[{sub_dir}] 处理完成(部分失败)，耗时: {elapsed_time:.2f}秒")
    return False


def main():
    """主函数"""
    validate_paths()
    
    sub_dirs = sorted(
        [d for d in os.listdir(PATH_CONFIG["input_dir"]) 
        if os.path.isdir(os.path.join(PATH_CONFIG["input_dir"], d)) and d.isdigit()],
        key=int
    )
    
    total_start = time.time()
    
    for sub_dir in sub_dirs:
        print(f"\n{'='*40}")
        print(f"处理目录: {sub_dir}")
        print(f"{'='*40}")
        
        if process_single_directory(sub_dir):
            print(f"[{sub_dir}] 处理成功")
        else:
            print(f"[{sub_dir}] 处理失败")

    total_elapsed = time.time() - total_start
    print("\n全流程完成！")
    print(f"总耗时: {total_elapsed:.2f}秒")
    print("结果位置：")
    print(f"中间文件: {PATH_CONFIG['output_dir']}")
    print(f"最终结果: 各子目录下的ct/model_ct.jpg")


if __name__ == "__main__":
    main()