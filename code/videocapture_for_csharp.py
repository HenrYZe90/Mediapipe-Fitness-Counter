import datetime
import os
import cv2
import numpy as np
import poseembedding as pe  # 姿态关键点编码模块
import poseclassifier as pc  # 姿态分类器
import resultsmooth as rs  # 分类结果平滑
import counter  # 动作计数器


def process(flag, pose_landmarks_input):
    # class_name需要与你的训练样本的两个动作状态图像文件夹的名字中的一个(或者是与fitness_poses_csvs_out中的一个csv文件的名字）保持一致，它后面将用于分类时的索引。
    # 具体是哪个动作文件夹的名字取决于你的运动是什么，例如：如果是深蹲，明显比较重要的判断计数动作是蹲下去；如果是引体向上，则判断计数的动作是向上拉到最高点的那个动作
    # class_name = 'squat_down'
    # out_video_path = 'squat-sample-out.mp4'
    if flag == 1:
        class_name = 'DeepSquat_down'
    elif flag == 2:
        class_name = 'HighKnees_prepare'

    # Initialize tracker, classifier and counter.
    # Do that before every video as all of them have state.

    # Folder with pose class CSVs. That should be the same folder you using while building classifier to output CSVs.
    pose_samples_folder = 'fitness_poses_csvs_out'

    # Initialize embedder.
    pose_embedder = pe.FullBodyPoseEmbedder()

    # Initialize classifier.
    # Check that you are using the same parameters as during bootstrapping.
    pose_classifier = pc.PoseClassifier(
        pose_samples_folder=pose_samples_folder,
        # class_name=class_name,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    # Initialize EMA smoothing.
    pose_classification_filter = rs.EMADictSmoothing(
        window_size=2,
        alpha=0.2)

    # Initialize counter.
    repetition_counter = counter.RepetitionCounter(
        flag=flag,
        class_name=class_name)

    # Run pose tracker.
    pose_landmarks = pose_landmarks_input

    if pose_landmarks is not None:
        # Get landmarks.
        assert pose_landmarks.shape == (33, 4), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

        # Classify the pose on the current frame.
        pose_classification = pose_classifier(pose_landmarks[:, :3])

        # Smooth classification using EMA.
        pose_classification_filtered = pose_classification_filter(pose_classification)
        print(datetime.datetime.now(), pose_classification_filtered)

        # Count repetitions.
        result = repetition_counter(pose_classification_filtered, pose_landmarks.landmark)
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
        repetitions_count = repetition_counter.n_repeats
