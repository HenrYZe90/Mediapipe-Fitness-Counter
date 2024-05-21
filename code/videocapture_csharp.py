import poseembedding as pe  # 姿态关键点编码模块
import poseclassifier as pc  # 姿态分类器
import resultsmooth as rs  # 分类结果平滑
import counter_csharp  # 动作计数器


def initialize_embedder():
    # 得到表示肢体动作的向量
    pose_embedder = pe.FullBodyPoseEmbedder()
    return pose_embedder


def initialize_classifier(pose_samples_folder, pose_embedder):
    # Check that you are using the same parameters as during bootstrapping.
    # 动作分类 动作A:2, 动作B:8 ...
    pose_classifier = pc.PoseClassifier(
        pose_samples_folder=pose_samples_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)
    return pose_classifier


def initialize_EMA_smoothing(window_size=1, alpha=0.5):
    # 平滑处理，过滤掉动作识别的毛刺，alpha为0~1之间，alpha越大，表示之前动作权重越小
    pose_classification_filter = rs.EMADictSmoothing(window_size, alpha)
    return pose_classification_filter


def initialize_counter(flag):
    # 计数+判断动作标准
    repetition_counter = counter_csharp.RepetitionCounter(flag=flag)
    return repetition_counter


def process(pose_landmarks, pose_classifier, pose_classification_filter, repetition_counter):
    if pose_landmarks is not None:
        # Get landmarks.
        assert pose_landmarks.shape == (33, 4), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

        # Classify the pose on the current frame.
        pose_classification = pose_classifier(pose_landmarks[:, :3])
        pose_classification = pose_classification_filter(pose_classification)

        # Count repetitions.
        counter_result = repetition_counter(pose_classification, pose_landmarks)

    return counter_result
