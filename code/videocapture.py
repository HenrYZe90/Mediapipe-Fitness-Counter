import datetime
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
# import tqdm
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
import mediapipe as mp
import poseembedding as pe  # 姿态关键点编码模块
import poseclassifier as pc  # 姿态分类器
import resultsmooth as rs  # 分类结果平滑
import counter  # 动作计数器
import visualizer as vs  # 可视化模块
from PIL import ImageDraw, ImageFont


def show_image(img, figsize=(10, 10)):
    """Shows output PIL image."""
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()


def process(flag):
    mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')
    if not os.path.exists('./video-output'):
        os.mkdir('./video-output')
    # class_name需要与你的训练样本的两个动作状态图像文件夹的名字中的一个(或者是与fitness_poses_csvs_out中的一个csv文件的名字）保持一致，它后面将用于分类时的索引。
    # 具体是哪个动作文件夹的名字取决于你的运动是什么，例如：如果是深蹲，明显比较重要的判断计数动作是蹲下去；如果是引体向上，则判断计数的动作是向上拉到最高点的那个动作
    # class_name = 'squat_down'
    # out_video_path = 'squat-sample-out.mp4'
    if flag == 1:
        class_name = 'DeepSquat_down'
        pose_samples_folder = './fitness_poses_csvs_out/DeepSquat'
    elif flag == 2:
        class_name = 'HighKnees_prepare'
        pose_samples_folder = './fitness_poses_csvs_out/HighKnees'
    elif flag == 3:
        class_name = 'SkippingRope_lowest'
        pose_samples_folder = './fitness_poses_csvs_out/SkippingRope'
    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    video_cap = cv2.VideoCapture(0)

    # Get some video parameters to generate output video with classification.
    # video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = 24
    video_width = 640
    video_height = 480

    # Initialize tracker, classifier and counter.
    # Do that before every video as all of them have state.

    # Folder with pose class CSVs. That should be the same folder you using while building classifier to output CSVs.
    # pose_samples_folder = 'fitness_poses_csvs_out'

    # 初始化，准备工作只做一次
    # Initialize tracker.
    # 得到关键点
    pose_tracker = mp_pose.Pose()

    # Initialize embedder.
    # 得到表示肢体动作的向量
    pose_embedder = pe.FullBodyPoseEmbedder()

    # Initialize classifier.
    # Check that you are using the same parameters as during bootstrapping.
    # 动作分类 动作A:2, 动作B:8 ...
    pose_classifier = pc.PoseClassifier(
        pose_samples_folder=pose_samples_folder,
        # class_name=class_name,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    # Uncomment to validate target poses used by classifier and find outliers.
    # outliers = pose_classifier.find_pose_sample_outliers()
    # print('Number of pose sample outliers (consider removing them): ', len(outliers))

    # Initialize EMA smoothing.
    # 平滑处理，过滤掉动作识别的毛刺
    pose_classification_filter = rs.EMADictSmoothing(
        window_size=2,
        alpha=0.2)

    # Initialize counter.
    # 计数+判断动作标准
    repetition_counter = counter.RepetitionCounter(
        flag=flag,
        class_name=class_name)

    # Initialize renderer.
    pose_classification_visualizer = vs.PoseClassificationVisualizer(
        class_name=class_name,
        # plot_x_max=100,  # 横轴长度
        # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
        plot_y_max=10)

    # Run classification on a video.

    # Open output video.
    # out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

    # 一帧一帧的处理
    # frame_idx = 0
    output_frame = None
    # with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:
    while video_cap.isOpened():
        # Get next frame of the video.
        success, input_frame = video_cap.read()
        if not success:
            break

        # Run pose tracker.
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        result = pose_tracker.process(image=input_frame)
        # 得到了骨骼点坐标
        pose_landmarks = result.pose_landmarks

        # Draw pose prediction.
        output_frame = input_frame.copy()
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)

        if pose_landmarks is not None:
            # Get landmarks.
            frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
            pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width, lmk.visibility]
                                       for lmk in pose_landmarks.landmark], dtype=np.float32)
            assert pose_landmarks.shape == (33, 4), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

            # Classify the pose on the current frame.
            pose_classification = pose_classifier(pose_landmarks[:, :3])

            # Smooth classification using EMA.
            pose_classification_filtered = pose_classification_filter(pose_classification)
            print(datetime.datetime.now(), pose_classification_filtered)

            # Count repetitions.
            result = repetition_counter(pose_classification_filtered, pose_landmarks)
            print(result)
            repetitions_count = result['n_repeat']
        else:
            # No pose => no classification on current frame.
            pose_classification = None

            # Still add empty classification to the filter to maintaining correct
            # smoothing for future frames.
            pose_classification_filtered = pose_classification_filter(dict())
            pose_classification_filtered = None

            # Don't update the counter presuming that person is 'frozen'. Just
            # take the latest repetitions count.
            repetitions_count = repetition_counter.n_repeat

        # Draw classification plot and repetition counter.
        output_frame = pose_classification_visualizer(
            frame=output_frame,
            pose_classification=pose_classification,
            pose_classification_filtered=pose_classification_filtered,
            repetitions_count=repetitions_count)

        # 实时输出检测画面
        cv2.imshow('video', cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))
        # Save the output frame.
        # out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))
        # 按键盘的q或者esc退出
        if cv2.waitKey(1) in [ord('q'), 27]:
            break

    # Close output video.
    # out_video.release()
    video_cap.release()
    cv2.destroyAllWindows()

    # Release MediaPipe resources.
    pose_tracker.close()

    # Show the last frame of the video.
    # if output_frame is not None:
    #     show_image(output_frame)
    # print(f"视频处理结束，输出保存在{out_video_path}")
