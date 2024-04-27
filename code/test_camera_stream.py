import videocapture as vc
import trainingsetprocess as tp

if __name__ == '__main__':
    # flag = 1 深蹲；flag = 2 提膝击掌
    flag = 1
    print("\n在视频窗口按英文状态下的q或esc退出摄像头采集")
    tp.trainset_process(flag)
    vc.process(flag)
