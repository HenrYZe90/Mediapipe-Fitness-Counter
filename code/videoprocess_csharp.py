import datetime
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
import tqdm
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
import poseembedding as pe  # 姿态关键点编码模块
import poseclassifier as pc  # 姿态分类器
import resultsmooth as rs  # 分类结果平滑
import counter  # 动作计数器
import visualizer as vs  # 可视化模块
from PIL import ImageDraw, ImageFont
import time
import logging


def show_image(img, figsize=(10, 10)):
    """Shows output PIL image."""
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()


def video_process(video_path, flag):
    mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')
    if not os.path.exists('./video-output'):
        os.mkdir('./video-output')
    # 指定视频路径和输出名称
    # video_path = 'pushup-sample.mp4'
    # class_name需要与你的训练样本的两个动作状态图像文件夹的名字中的一个（或者是与fitness_poses_csvs_out中的一个csv文件的名字）保持一致，它后面将用于分类时的索引。
    # 具体是哪个动作文件夹的名字取决于你的运动是什么，例如：如果是深蹲，明显比较重要的判断计数动作是蹲下去；如果是引体向上，则判断计数的动作是向上拉到最高点的那个动作；如果是俯卧撑，则判断计数的动作是最低点的那个动作
    if flag == 1:
        class_name = 'DeepSquat_down'
        out_video_path = './video-output/' + class_name.split('_')[0] + ' ' + mkfile_time + '.mp4'
        pose_samples_folder = './fitness_poses_csvs_out/DeepSquat'
    elif flag == 2:
        class_name = 'HighKnees_prepare'
        out_video_path = './video-output/' + class_name.split('_')[0] + ' ' + mkfile_time + '.mp4'
        pose_samples_folder = './fitness_poses_csvs_out/HighKnees'
    elif flag == 3:
        class_name = 'SkippingRope_lowest'
        out_video_path = './video-output/' + class_name.split('_')[0] + ' ' + mkfile_time + '.mp4'
        pose_samples_folder = './fitness_poses_csvs_out/SkippingRope'

    # 配置logging模块
    logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info('0 - start counting...')

    # Open the video.
    video_cap = cv2.VideoCapture(video_path)

    # Get some video parameters to generate output video with classification.
    video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize tracker, classifier and counter.
    # Do that before every video as all of them have state.

    # Folder with pose class CSVs. That should be the same folder you using while building classifier to output CSVs.
    # pose_samples_folder = 'fitness_poses_csvs_out'

    # Initialize tracker.
    pose_tracker = mp_pose.Pose()

    counter_result = None
    while True:
        # Get next frame of the video.
        success, input_frame = video_cap.read()
        if not success:
            break

        start_time = time.time()
        # Initialize embedder.
        pose_embedder = pe.FullBodyPoseEmbedder()
        print(f"耗时分析 embedder 初始化 {time.time() - start_time} sec.")
        logging.info('1 - pose embedder initialized')

        start_time = time.time()
        # Initialize classifier.
        # Check that you are using the same parameters as during bootstrapping.
        pose_classifier = pc.PoseClassifier(
            pose_samples_folder=pose_samples_folder,
            pose_embedder=pose_embedder,
            top_n_by_max_distance=30,
            top_n_by_mean_distance=10)
        print(f"耗时分析 classifier 初始化 {time.time() - start_time} sec.")
        logging.info('2 - pose classifier initialized')

        start_time = time.time()
        # Initialize counter.
        repetition_counter = counter.RepetitionCounter(
            flag=flag,
            class_name=class_name,
            prev_result=counter_result)
        print(f"耗时分析 counter 初始化 {time.time() - start_time} sec.")
        logging.info('3 - repetition counter initialized')

        start_time = time.time()
        # Run pose tracker.
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        result = pose_tracker.process(image=input_frame)
        pose_landmarks = result.pose_landmarks
        print(f"耗时分析 获取肢体关键点 {time.time() - start_time} sec.")
        logging.info('4 - pose landmarks to numpy array done')

        if pose_landmarks is None:
            continue

        start_time = time.time()
        if pose_landmarks is not None:
            # Get landmarks.
            frame_height, frame_width = input_frame.shape[0], input_frame.shape[1]
            pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width, lmk.visibility]
                                       for lmk in pose_landmarks.landmark], dtype=np.float32)
            assert pose_landmarks.shape == (33, 4), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

            # Classify the pose on the current frame.
            pose_classification = pose_classifier(pose_landmarks[:, :3])
            # print('完成时间 ', datetime.datetime.now())
            # print('动作分类 ', pose_classification)
            logging.info('5 - pose classification done')

            # Count repetitions.
            counter_result = repetition_counter(pose_classification, pose_landmarks)
            # print('输出结果 ', counter_result)
            logging.info('6 - returned result')

        # print(f"耗时分析 计算结果 {time.time() - start_time} sec.")
        # print()
