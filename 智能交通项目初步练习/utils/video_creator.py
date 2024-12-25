import cv2
import os


def images_to_video(image_folder, output_video_path, fps=30):
    # 获取图片文件列表，假设图片是按顺序命名的，例如 img001.jpg, img002.jpg, ...
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))])

    # 获取第一张图片的尺寸，用于设置视频的尺寸
    frame = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, layers = frame.shape

    # 定义视频编解码器并创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编解码器
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image_file in image_files:
        # 读取图片
        img_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(img_path)

        # 写入视频
        out.write(frame)

    # 释放资源
    out.release()
    print(f"Video saved to {output_video_path}")


# 示例用法
image_folder = '../data/MVI_39511'  # 替换为你的图片文件夹路径
output_video_path = '../data/output_video.mp4'  # 输出视频文件的路径
fps = 30  # 每秒帧数，根据你的需求调整

images_to_video(image_folder, output_video_path, fps)
