import cv2
import os


def extract_frames(video_path, output_folder, filename):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

        # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查是否成功打开视频文件
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

        # 初始化帧计数器
    frame_count = 0

    # 逐帧读取视频
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

            # 保存帧到文件
        frame_path = os.path.join(output_folder, f'{filename}_frame_{frame_count:04d}.jpg')  # 使用四位数编号，并保存为jpg格式
        cv2.imwrite(frame_path, frame)

        # 递增帧计数器
        frame_count += 1

        # 释放视频文件和所有OpenCV窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = r'.\video-sample\situp'
    filenames = os.listdir(video_path)
    for filename in filenames:
        output_folder = 'frames'  # 输出文件夹名，将在这个文件夹下保存每一帧
        extract_frames(os.path.join(video_path, filename), output_folder, filename.split('.')[0])
