import os 
import sys

import numpy as np
import cv2
import tensorflow as tf
# import tensorflow.gfile as gfile

def create_video_lists(video_dir):
    if not tf.gfile.Exists(video_dir):
        tf.logging.error("Video directory '" + video_dir + "' not found. ")
        return None

    results = dict()
    sub_dirs = [x[0] for x in tf.gfile.Walk(video_dir)]
    # print(sub_dirs)
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['avi']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == video_dir:
            continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(video_dir, dir_name, '*.' + extension)
            file_list.extend(tf.gfile.Glob(file_glob))
        results[dir_name] = file_list
        # if not file_list:
        #     tf.logging.warning('No files found')
        #     continue
        # if len(file_list) < 100:
        #     tf.logging.warning(
        #         'WARNING: Folder has less than 100 videos, which may cause issues.')
        # elif len(file_list) > 100:
        #     tf.logging.warning(
        #         'WARNING: Folder {} has more than {} images. Some images will '
        #         'never be selected.'.format(dir_name, 100))
    return results

def extract(video_path, sub_image_dir): 
    cap = cv2.VideoCapture(video_path)
    
    cnt = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True: 
            cv2.imwrite(os.path.join(sub_image_dir, '%d.png'%(cnt)), frame)
            cnt += 1
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break
    cap.release()
    return cnt

if __name__ == '__main__': 
    video_dir = '../hmdb51_org'
    image_dir = '../hmdb51_org_images'

    pid = int(sys.argv[1])
    tid = int(sys.argv[2])

    video_list = create_video_lists(video_dir) 
    # print(video_list)
    cnt_list = []
    count = 0
    for class_name, sub_video_list in video_list.items(): 
        for video_path in sub_video_list:
            if count % tid == pid:
                print(video_path)
                sub_image_dir = video_path.replace(os.path.basename(video_dir), os.path.basename(image_dir))
                if not os.path.exists(sub_image_dir): os.makedirs(sub_image_dir)
                print(sub_image_dir)
                cnt = extract(video_path, sub_image_dir)
                cnt_list.append(cnt)
            count += 1
    cn_list = np.array(cnt_list).reshape((-1,1)).astype(int)
    print(cn_list.min(), cn_list.max(), cn_list.mean())
    np.savetxt('hmdb51_frames_stats_%d.txt'%(pid), cnt_list, fmt='%d')
    