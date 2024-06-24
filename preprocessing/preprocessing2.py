import os
import cv2
import numpy as np

from ultralytics import YOLO

def face_matrix_yolo(video_path,model,output_dir,resize_value=0):
    cap = cv2.VideoCapture(video_path)

    int_box = None
    size = None
    frame_count = 0
    cropped_faces = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    while cap.isOpened():
        ret, frame = cap.read()
        frame=cv2.resize(frame,(960,540))
        if not ret:
            break

        results = model(frame)
        # x1, y1, x2, y2, confidence, class
        detections = results[0].boxes.data

        if len(detections) > 0:
            for d in detections:
                x1,y1,x2,y2,conf,cls = d
                i = int(cls)
                x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])

                if model.names[i] == "face":
                    if int_box is None:
                        int_box = (x1,y1,x2,y2)
                        w = 68
                        h = 172
                        size = int((w+h)/2) + 20 #Added padding
                    #Center coordinates
                    c_x = x1+(x2-x1)//2 -30
                    c_y = y1+(y2-y1)//2 -50

                    n_x = max(0, c_x-size//2)
                    n_y = max(0,c_y-size//2)
                    n_w = int(1.75*(w + 40))
                    n_h = int(1.75*(h))
                    #Cropping Face Region
                    face = frame[n_y:n_y + n_h,n_x:n_x+n_w]
                    cropped_faces.append(face)

                    #Save the cropped face images

                    save_cropped_frames(face,output_dir,frame_count)
                    frame_count+=1
                    #Now we have the cropped face matrix, we can resize the face matrix and send it to the model
                    cv2.rectangle(frame, (n_x, n_y), (n_x + n_w, n_y + n_h), (0, 255, 0), 2)
                    
        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    # save_vid_folder = os.path.join(output_dir,'CroppedVideo')
    # if not os.path.exists(save_vid_folder):
    #     os.makedirs(save_vid_folder)
    # save_cropped_faces_as_video(cropped_faces,os.path.join(save_vid_folder,'cropped_faces_video.mp4'))

    # normalized_frames_folder = os.path.join(output_dir,'NormalizedFrames')
    # if not os.path.exists(normalized_frames_folder):
    #     os.makedirs(normalized_frames_folder)
    # normalized_diffs = calculate_normalized_difference(cropped_faces)
    # for i,diff in enumerate(normalized_diffs):
    #     cv2.imwrite(os.path.join(normalized_frames_folder,f'normalized_diff_{i:04d}.png'),diff)
    # print(size)
    # print(face)

#Assuming the bounding box is to be a square

def resize_face_matrix(face,resize_value):
    face_resized = cv2.resize(face,(resize_value,resize_value))
    return face_resized

def save_cropped_frames(face,ouput_dir,frame_count):
    frames_folder = os.path.join(ouput_dir,"Frames")
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)
    filename = os.path.join(frames_folder,f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(filename,face)

def save_cropped_faces_as_video(cropped_faces, output_video_path, fps=30):
    if not cropped_faces:
        raise ValueError("No cropped faces to save.")

    height, width, _ = cropped_faces[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in cropped_faces:
        out.write(frame)

    out.release()

# def calculate_normalized_difference(cropped_faces):
#     normalized_diffs = []

#     for i in range(1, len(cropped_faces)):
#         c_t1 = cropped_faces[i].astype(np.float32)
#         c_t = cropped_faces[i - 1].astype(np.float32)

#         diff = (c_t1 - c_t) / (c_t1 + c_t + 1e-5)  # Add small value to avoid division by zero
#         norm_diff = np.clip(diff, -3 * np.std(diff), 3 * np.std(diff))
#         norm_diff = ((norm_diff - norm_diff.min()) / (norm_diff.max() - norm_diff.min()) * 255).astype(np.uint8)
#         normalized_diffs.append(norm_diff)
#     return normalized_diffs

model = YOLO('yolov8n-face.pt')
root = 'DATASET_2'
dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

video_path = '/home/uas-dtu/Downloads/case14_10sec.avi'
output_dir = "generated_dataset_test"
face_matrix_yolo(video_path,model,output_dir)

