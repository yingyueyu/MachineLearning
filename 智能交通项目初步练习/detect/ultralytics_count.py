import cv2

"""
区域计数
"""
from ultralytics import solutions


def count_objects_in_region(video_path, output_video_path, model_path):
    """Count objects in a specific region within a video."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # 指定计数区域
    region_points = [[373, 380], [337, 446], [867, 446], [784, 380]]
    # 目标在指定区域中出现并计数
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_path)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        im0 = counter.count(im0)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


# 作者在底层代码中，是通过流数据（与我们的一帧一帧监测图像不同）进行监测目标的
count_objects_in_region("../data/output_video.mp4", "output_video.avi", "../weights/yolo11n.pt")
