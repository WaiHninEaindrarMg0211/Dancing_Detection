import os
import numpy as np
import glob, re, time
import ipywidgets as widgets
import cv2 as cv
import torch, cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

frameSkip = 15 
desired_width = 2000  
desired_height = 700 
min_area = 10000 

yolov8_config ="./custom_data.yaml"
yolov8_weights ="./detection_model/best.pt"
videos_path = './dancing.mp4'

bbox_list = []
start_time = time.time()

cap = cv2.VideoCapture(videos_path)
f_cnt = cv2.CAP_PROP_FRAME_COUNT
numFrames = int(cap.get(f_cnt))
count = 1

save_path = "./output/"
resultVdoName = './output/output_video.mp4'
out = cv2.VideoWriter(resultVdoName, cv2.VideoWriter_fourcc(*'mp4v'), 5, (desired_width, desired_height))          

for i in range(0,numFrames,frameSkip):
    
    cap.set(cv2.CAP_PROP_POS_FRAMES,i)
    ret,frame = cap.read()
    
    if not ret:
        print("Can't read the frame")
        break
 
    img = frame
    ori_img = img
    img = cv2.resize(img, (1080, 1080))
    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

    # detection
    detector = YOLO(yolov8_weights)
    outputs = detector.predict(img)

    bboxs = []
    scores = []
    class_ids = []
    masks = []

    filtered_bboxs = []
    filtered_scores = []
    filtered_class_ids = []
    filtered_masks_array = []

    for r in outputs:

        bboxs.extend(r.boxes.xyxy)
        scores.extend(r.boxes.conf)
        class_ids.extend(r.boxes.cls)
        masks.extend(r.masks.data)
    
    bboxs = np.array([bbox.cpu().numpy() for bbox in bboxs])
    scores = np.array([score.cpu().numpy() for score in scores])
    class_ids = np.array([class_id.cpu().numpy() for class_id in class_ids]) 

    masks_array = np.array([mask.cpu().numpy() for mask in masks])
    masks = np.array(masks_array)
    mask_sum = masks.sum(axis=0)
    mask_sum[mask_sum > 1] = 1
    mask_sum = (mask_sum*255).astype(np.uint8)

    resized_mask = cv2.resize(mask_sum, (img.shape[1], img.shape[0]))
    color_mask = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
    color_mask[np.where((color_mask == [255, 255, 255]).all(axis=2))] = [255, 0, 0] 

    for idx, obj_mask in enumerate(masks_array):
        area = np.count_nonzero(obj_mask)

        if area > min_area:
            filtered_bboxs.append(bboxs[idx])
            filtered_scores.append(scores[idx])
            filtered_class_ids.append(class_ids[idx])
            filtered_masks_array.append(obj_mask)

    # Convert filtered results to numpy arrays
    filtered_bboxs = np.array(filtered_bboxs)
    filtered_scores = np.array(filtered_scores)
    filtered_class_ids = np.array(filtered_class_ids)
    filtered_masks_array = np.array(filtered_masks_array)


    # Draw bounding boxes and labels on the original image
    for bbox, score, class_id, mask in zip(filtered_bboxs, filtered_scores, filtered_class_ids, filtered_masks_array):
        x1, y1, x2, y2 = bbox.astype(int)
        label = 'Person'
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    result = cv2.addWeighted(img, 1, color_mask, 0.6, 0)


    replicated_mask = np.repeat(mask_sum[:, :, np.newaxis], 3, axis=2)
    replicated_mask_ = cv2.resize(replicated_mask, (1080, 1080))

    display_image = np.hstack((ori_img, result, replicated_mask_))
    final_image = cv2.resize(display_image, (desired_width, desired_height))


    # Add a final label to the image
    title_label = "Original vs Detection and Segmentation vs Binary Mask"
    cv2.putText(final_image, title_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    save_video_path = (os.path.join(save_path, '{}_{}.jpg'.format("dancing", count)))
    cv2.imshow("Detection", final_image)
    cv2.imwrite(save_video_path, final_image)
    out.write(final_image)
    count = count + 1

    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

out.release()
end_time = time.time()
test_duration = end_time - start_time

hours, remainder = divmod(test_duration, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Testing Duration : {int(hours):02d} : {int(minutes):02d} : {int(seconds):02d}")
cv2.destroyAllWindows()


 






