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
    elif flag == 4:
        class_name = 'PushUp_down'
        out_video_path = './video-output/' + class_name.split('_')[0] + ' ' + mkfile_time + '.mp4'
        pose_samples_folder = './fitness_poses_csvs_out/PushUp'
    elif flag == 5:
        class_name = 'SitUp_down'
        out_video_path = './video-output/' + class_name.split('_')[0] + ' ' + mkfile_time + '.mp4'
        pose_samples_folder = './fitness_poses_csvs_out/SitUp'

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

    # Initialize embedder.
    pose_embedder = pe.FullBodyPoseEmbedder()

    # Initialize classifier.
    # Check that you are using the same parameters as during bootstrapping.
    pose_classifier = pc.PoseClassifier(
        pose_samples_folder=pose_samples_folder,
        pose_embedder=pose_embedder,
        # class_name=class_name,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    # Uncomment to validate target poses used by classifier and find outliers.
    # outliers = pose_classifier.find_pose_sample_outliers()
    # print('Number of pose sample outliers (consider removing them): ', len(outliers))

    # Initialize EMA smoothing.
    pose_classification_filter = rs.EMADictSmoothing(
        window_size=1,
        alpha=0.3)

    # Initialize counter.
    repetition_counter = counter.RepetitionCounter(
        flag=flag,
        enter_threshold=8,
        exit_threshold=2)

    # Initialize renderer.
    pose_classification_visualizer = vs.PoseClassificationVisualizer(
        class_name=class_name,
        plot_x_max=video_n_frames,
        # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
        plot_y_max=10)

    # Run classification on a video.

    # Open output video.
    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

    frame_idx = 0
    output_frame = None
    with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:
        while True:
            # Get next frame of the video.
            success, input_frame = video_cap.read()
            if not success:
                break

            # Run pose tracker.
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            result = pose_tracker.process(image=input_frame)
            pose_landmarks = result.pose_landmarks

            if pose_landmarks is None:
                continue

            # Draw pose prediction.
            output_frame = input_frame.copy()
            if pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS)

            if pose_landmarks is not None:
                # print(pose_landmarks)
                # print(pose_landmarks.landmark)

                # Get landmarks.
                frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width, lmk.visibility]
                                           for lmk in pose_landmarks.landmark], dtype=np.float32)
                assert pose_landmarks.shape == (33, 4), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
                # 设置打印选项以抑制科学计数法
                np.set_printoptions(suppress=True, precision=2)  # precision参数控制小数点后的位数
                # print(pose_landmarks)

                # Classify the pose on the current frame.
                pose_classification = pose_classifier(pose_landmarks[:, :3])
                # print(pose_classification)

                # Smooth classification using EMA.
                pose_classification_filtered = pose_classification_filter(pose_classification)
                print(datetime.datetime.now(), pose_classification_filtered)

                # Count repetitions.
                result = repetition_counter(pose_classification_filtered, pose_landmarks)
                print(result)
                # repetitions_count = result['n_repeat']
                repetitions_count = repetition_counter.n_repeats

            else:
                # No pose => no classification on current frame.
                pose_classification = None

                # Still add empty classification to the filter to maintain correct
                # smoothing for future frames.
                pose_classification_filtered = pose_classification_filter(dict())
                pose_classification_filtered = None

                # Don't update the counter presuming that person is 'frozen'. Just
                # take the latest repetitions count.
                repetitions_count = repetition_counter.n_repeats

            # Draw classification plot and repetition counter.
            output_frame = pose_classification_visualizer(
                frame=output_frame,
                pose_classification=pose_classification,
                pose_classification_filtered=pose_classification_filtered,
                repetitions_count=repetitions_count)

            # Save the output frame.
            out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))
            frame_idx += 1
            pbar.update()

            # import time
            # current_time = time.time() * 1000
            # cv2.imwrite(os.path.join('output', f'{current_time}.jpg'), np.array(output_frame))

            # angle_result = str(int(result['shoulder_wrist_angle_left'])) + '_' + str(int(result['shoulder_wrist_angle_right'])) + '_' + str(int(result['shoulder_elbow_wrist_left'])) + '_' + str(
            #     int(result['shoulder_elbow_wrist_right']))
            # cv2.imwrite(os.path.join('output', f'{angle_result}.jpg'), np.array(output_frame))

    # Close output video.
    out_video.release()

    # Release MediaPipe resources.
    pose_tracker.close()

    # Show the last frame of the video.
    # if output_frame is not None:
    #     show_image(output_frame)
    print(f"视频处理结束，输出保存在{out_video_path}")
