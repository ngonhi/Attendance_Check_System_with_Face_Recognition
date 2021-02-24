# Code to generate faces from multiple videos 
import sys
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from mtcnn.mtcnn import MTCNN
from imutils import paths
import face_preprocess
import numpy as np
import argparse
import cv2
import os
import json



def load_video(path):
    '''
    Load video path
    path: folder containing all videos

    Returns a list of video path
    '''
    videos = [os.path.join(path, video) for video in os.listdir(path)]
    return videos

def load_json(filePath):
    '''
    Load user information
    '''
    with open(filePath, 'r', encoding="utf-8") as f:
        return json.load(f)

def get_faces(videos, detector, user_info, max_faces=30):
    for video in videos:
        video_name = os.path.basename(video)
        key = video_name.split('.')[0]
        emp_id = user_info[key]['emp_id']
        emp_name = user_info[key]['name']
        print(emp_name, emp_id)
        if os.path.exists(os.path.join('./dataset', emp_id)):
            continue
        faces = 0
        frames = 0
        max_bbox = np.zeros(4)

        cap = cv2.VideoCapture(video)
        while faces < max_faces:
            ret, frame = cap.read()
            frames += 1

            if frames % 2 == 0:
                # Get all faces on current frame
                bboxes = detector.detect_faces(frame)
                
                if len(bboxes) != 0:
                # Get only the biggest face
                    max_area = 0
                    for bboxe in bboxes:
                        bbox = bboxe["box"]
                        bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                        keypoints = bboxe["keypoints"]
                        area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                    # if area > max_area:
                    #     max_bbox = bbox
                        landmarks = keypoints
                    #     max_area = area

                    # max_bbox = max_bbox[0:4]

                    # convert to face_preprocess.preprocess input
                        landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                     landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                        landmarks = landmarks.reshape((2,5)).T
                        nimg = face_preprocess.preprocess(frame, bbox, landmarks, image_size='112,112')

                        output_path = os.path.join('./dataset', emp_id)
                        if not(os.path.exists(output_path)):
                            os.makedirs(output_path)
                        cv2.imwrite(os.path.join(output_path, "{}.jpg".format(faces+1)), nimg)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                        print("[INFO] {} faces detected".format(faces+1))
                        faces += 1
                cv2.imshow("Face detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True,
                help="Path to folder containing video")
    args = vars(ap.parse_args())

    videos = load_video(args['video'])
    user_info = load_json('user_info.json')
    detector = MTCNN()

    print('Start extracting faces from videos')
    get_faces(videos, detector, user_info)
    print('Finished extracting faces from videos')
    return


if __name__ == '__main__':
    main()
