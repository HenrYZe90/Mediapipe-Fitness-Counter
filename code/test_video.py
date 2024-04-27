import videoprocess as vp
import trainingsetprocess as tp

if __name__ == '__main__':
    # flag = 1 深蹲；flag = 2 提膝击掌
    flag = 2
    # 输入要处理的视频的路径
    video_path = ".\\video-sample\\HighKnees02.mp4"
    tp.trainset_process(flag)
    vp.video_process(video_path, flag)
