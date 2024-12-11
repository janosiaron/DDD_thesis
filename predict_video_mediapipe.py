import time
import os
import argparse
import glob
import numpy as np
import cv2
import dlib
import torch
import statistics
import keras
from calculate_iou import calculate_iou
from matplotlib import pylab as plt
import mediapipe as mp
from scipy.spatial import distance as dist
import logging
from datetime import datetime
import matplotlib
from playsound import playsound

# Crops out the necessary part from the specified frame
def get_prediction_image(eye_x, eye_y, pred_image):
    min_x = int(min(eye_x))
    max_x = int(max(eye_x))
    min_y = int(min(eye_y))
    max_y = int(max(eye_y))
    pred_image = pred_image[round(min_y * 0.95):round(max_y * 1.05),
                            round(min_x * 0.95):round(max_x * 1.05)]

    pred_image = cv2.resize(pred_image, (80, 80))
    pred_image = cv2.cvtColor(pred_image, cv2.COLOR_RGB2GRAY)
    pred_image = pred_image / 255.0
    return pred_image

# Returns the relevant points from the mesh
def get_mesh_points_and_indexes(image, facemesh, landmarks, indexes):
    source_indexes = []
    relative_source_points = []
    for source_idx, target_idx in facemesh:
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]

        relative_source = (int(source.x * image.shape[1]), int(source.y * image.shape[0]))
        relative_target = (int(target.x * image.shape[1]), int(target.y * image.shape[0]))
        source_indexes.append(source_idx)

        if indexes.__contains__(source_idx):
            relative_source_points.append(relative_source)
    return source_indexes, relative_source_points

# Calculates the Euclidean distances between the given points
def get_euclidean_distances(points, indexes):
    distances = []
    for first, second in indexes:
        distances.append(dist.euclidean(points[first], points[second]))
    return distances

# Checks if based on the last X seconds drowsiness occurred or not
def check_drowsiness(calc_blink_durations, calc_blink_times, calc_yawn_durations, calc_yawn_times, number_of_frames_in_red):
    is_drowsy = False
    # Checks the nodding events
    if number_of_frames_in_red >= 20:
        is_drowsy = True
    else:
        current_time = time.perf_counter_ns()
        found_time_blink = False
        # Checks if a blink event happened in the last X seconds
        for i in range(len(calc_blink_times)):
            if calc_blink_times[i] > (current_time - 60000000):
                found_time_blink = True
                break
        found_time_yawn = False
        # Checks if a yawn event happened in the last X seconds
        for j in range(len(calc_yawn_times)):
            if calc_yawn_times[j] > (current_time - 60000000):
                found_time_yawn = True
                break
        # Checks if the PERCLOS value is bigger than the threshold
        if found_time_blink:
            if sum(calc_blink_durations[i:]) > (1800 * 0.24):
                is_drowsy = True
        # Checks if the FOM value is bigger than the threshold
        if found_time_yawn:
            if sum(calc_yawn_durations[j:]) > (1800 * 0.16):
                is_drowsy = True

    return is_drowsy


# img_dir = "" # Enter Directory of all images

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

failed_no_face = 0
failed_bcuz_iou = 0
successful = 0
first_frame = True
numberOfFramesInYellow = 0
numberOfFramesInRed = 0
numberOfFramesGoingBack = 0
gone_red_without_going_back_up = False
underYellowLine = False
underRedLine = False
noOfBlinks = 0
noOfNods = 0
prevState = 0  # 0: Default, 1: Yellow Zone, 2: Red Zone
eventOngoing = False
blinkDuration = 0
predictions = []
MOUTH_INDEXES = [185, 409, 37, 84, 269, 17, 0, 314, 267, 405, 39, 181]
LEFT_EYE_INDEXES = [384, 385, 386, 387, 388, 381, 380, 374, 373, 390]  # side: 398, 263
RIGHT_EYE_INDEXES = [161, 160, 159, 158, 157, 163, 144, 145, 153, 154]  # side: 33, 173
MOUTH_PAIRS = [[10, 11], [0, 1], [3, 4], [5, 6], [2, 8], [2, 7]]  # the last one is the side of the mouth across
RIGHT_EYE_PAIRS = [[4, 6], [2, 8], [9, 1], [3, 5], [7, 0]]
LEFT_EYE_PAIRS = [[5, 6], [2, 9], [0, 7], [3, 4], [1, 8]]
BLINK_THRESHOLD = 1
YAWN_THRESHOLD = 60
prediction_batch_eye = []
prediction_batch_mouth = []
prediction_eye = [[-1]]
prediction_mouth = [[-1]]
rolling_data_eye = []
rolling_data_mouth = []
blink_durations_calculated = []
blink_durations_predicted = []
yawn_durations = []
eyes_closed_consecutively = 0
yawning_consecutively = 0
blinks_calculated = []
blinks_predicted = []
yawns_predicted = []
yawn_durations_predicted = []
yawns = []
first_thirty_frame = True
first_thirty_frame_points_yellow = []
first_thirty_frame_points_red = []
first_thirty_frame_counter = 0
set_lines = False
continue_blink = False
prediction_blink_counter = 0
prediction_yawn_counter = 0
NUMBER_OF_ZEROES_TO_IGNORE = 1
previous_blink_count = 0
MAR_values = []
MAR_threshold_values = []
EAR_values = []
EAR_threshold_values = []
nodding_values = []
mouth_prediction_values = []
ear_prediction_values = []
failed_to_detect_face = False
face_detected_again_counter = 0

# Initialize a logger
logger = logging.getLogger("Logger")
log_filename = "logs\\" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ".log"
date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
logging.basicConfig(filename=f"logs\{date}.log", encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s')
# The path to the video to be analysed
# ----------------------------------------------------------------------------------------------------------------------------------------
filename = r"test" # |
# ----------------------------------------------------------------------------------------------------------------------------------------

logger.debug("\n------------------------------------------------------------------------------------------------------")
logger.debug("\nThe analysed video is: %s", filename)
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("test_full_frontal.mp4")
ddd_name = "_".join(filename.split(os.sep)[-4:-1])
output_destination = f"videos\\DDD_training_{ddd_name}_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
result_folder = f"results\\DDD_training_{ddd_name}" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "\\"
os.makedirs(result_folder, exist_ok=True)

video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video = cv2.VideoWriter(output_destination, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30,
                               (video_width, video_height)) # width - height
logger.debug("Output video location: %s", output_destination)

blink_model = keras.saving.load_model(
    "trained_models\\retrained_model6\\retrained_classifier.keras")

yawn_model = keras.saving.load_model(
    "trained_models\yawn_model1\yawn_classifier.keras"
)

continuous_closed_eye = 0
previous_prediction = 0
time_array = []
mp_process_time_array = []
model_predict_time_array = []
idx = 0

start_time = time.time()

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    if not cap.isOpened():
        print("Error reading video file")
    while cap.isOpened():
        ret, frame = cap.read()
        frame_start = time.time()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            mp_process_start = time.time()
            results = face_mesh.process(image)
            mp_process_end = time.time()
            mp_process_time_array.append(mp_process_end - mp_process_start)
            height = frame.shape[0]
            width = frame.shape[1]
            # Mediapipe did not find a face in the image
            if not results.multi_face_landmarks:
                if underRedLine:
                    numberOfFramesInRed += 1
                    nodding_values.append(1)
                else:
                    nodding_values.append(0)
                failed_no_face += 1
                idx += 1
                failed_to_detect_face = True
                face_detected_again_counter = 0
                # cv2.imwrite(result_folder + f'\\{idx}.png', image)
                output_video.write(image)
                print("No face detected")
            # Mediapipe found a face in the image
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]

                source_left_eye_indexes, relative_source_left_eye_points = get_mesh_points_and_indexes(image,
                                                                                                       mp_face_mesh.FACEMESH_LEFT_EYE,
                                                                                                       landmarks, LEFT_EYE_INDEXES)
                source_right_eye_indexes, relative_source_right_eye_points = get_mesh_points_and_indexes(image,
                                                                                                         mp_face_mesh.FACEMESH_RIGHT_EYE,
                                                                                                         landmarks, RIGHT_EYE_INDEXES)
                source_mouth_indexes, relative_source_mouth_points = get_mesh_points_and_indexes(image,
                                                                                                 mp_face_mesh.FACEMESH_LIPS,
                                                                                                 landmarks, MOUTH_INDEXES)

                # Get the distances between the designated pairs
                mouth_distances = get_euclidean_distances(relative_source_mouth_points, MOUTH_PAIRS)
                right_eye_distances = get_euclidean_distances(relative_source_right_eye_points, RIGHT_EYE_PAIRS)
                left_eye_distances = get_euclidean_distances(relative_source_left_eye_points, LEFT_EYE_PAIRS)

                mean_distance_mouth = sum(mouth_distances[0:-1]) / (len(mouth_distances[0:-1]) * mouth_distances[-1])
                mean_distance_right_eye = np.mean(right_eye_distances)
                mean_distance_left_eye = np.mean(left_eye_distances)
                mean_distance_eyes = (mean_distance_left_eye + mean_distance_right_eye) / 2

                #cv2.imshow("frame", frame)
                """for i in range(len(relative_source_left_eye_points)):
                    cv2.circle(frame,relative_source_left_eye_points[i],radius=2,color=(0,255,255),thickness=2)
                    #cv2.putText(frame,f'{source_left_eye_indexes[i]}',relative_source_left_eye_points[i], cv2.FONT_HERSHEY_COMPLEX, 0.3, (255,0,0), 1)
                for i in range(len(relative_source_right_eye_points)):
                    cv2.circle(frame,relative_source_right_eye_points[i],radius=2,color=(0,255,255),thickness=2)
                    #cv2.putText(frame,f'{source_right_eye_indexes[i]}',relative_source_right_eye_points[i], cv2.FONT_HERSHEY_COMPLEX, 0.3, (255,0,0), 1)"""

                # Rolling data collection and event saving for eyes
                if len(rolling_data_eye) < 1:
                    rolling_data_eye.append(mean_distance_eyes)
                else:
                    lower_bound_eye = 0.95 * eye_threshold
                    upper_bound_eye = 1.05 * eye_threshold
                    if lower_bound_eye <= mean_distance_eyes <= upper_bound_eye:
                        if len(rolling_data_eye) < 150:
                            rolling_data_eye.append(mean_distance_eyes)
                        else:
                            rolling_data_eye = rolling_data_eye[1:]
                            rolling_data_eye.append(mean_distance_eyes)


                eye_threshold = np.mean(rolling_data_eye)

                EAR_values.append(round(mean_distance_eyes, 2))
                EAR_threshold_values.append(round(eye_threshold * 0.8, 2))

                if mean_distance_eyes < eye_threshold * 0.8:
                    eyes_closed_consecutively += 1
                elif eyes_closed_consecutively > BLINK_THRESHOLD:
                    blinks_calculated.append(time.perf_counter_ns())
                    blink_durations_calculated.append(eyes_closed_consecutively)
                    """cv2.putText(img_copy, f"BLINK HAPPENED (CALCULATED)!", (0, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)"""
                    logger.debug("Blink happened (calculated), duration: %f", eyes_closed_consecutively)
                    eyes_closed_consecutively = 0

                # Rolling data collection and event saving for mouth
                if len(rolling_data_mouth) < 1:
                    rolling_data_mouth.append(mean_distance_mouth)
                else:
                    lower_bound_mouth = 0.95 * mouth_threshold
                    upper_bound_mouth = 1.05 * mouth_threshold
                    if lower_bound_mouth <= mean_distance_mouth <= upper_bound_mouth:
                        if len(rolling_data_mouth) < 500:
                            rolling_data_mouth.append(mean_distance_mouth)
                        else:
                            rolling_data_mouth = rolling_data_mouth[1:]
                            rolling_data_mouth.append(mean_distance_mouth)

                mouth_threshold = np.mean(rolling_data_mouth)
                yawn_threshold = mouth_threshold * 1.25
                MAR_values.append(round(mean_distance_mouth, 2))
                MAR_threshold_values.append(round(yawn_threshold, 2))

                if mean_distance_mouth > yawn_threshold:
                    yawning_consecutively += 1
                elif yawning_consecutively >= YAWN_THRESHOLD:
                    yawns.append(time.perf_counter_ns())
                    yawn_durations.append(yawning_consecutively)
                    logger.debug("Yawn happened, duration: %f", yawning_consecutively)
                    yawning_consecutively = 0

                """for i in range(len(relative_source_right_eye_points)):
                    cv2.circle(frame, relative_source_right_eye_points[i], radius=2, color=(0, 255, 255), thickness=2)
                    cv2.circle(frame, relative_source_left_eye_points[i], radius=2, color=(0, 255, 255), thickness=2)"""
                    #cv2.putText(frame, f'{source_mouth_indexes[i]}', relative_source_mouth_points[i],
                                #cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 0), 1)

                img_copy = frame.copy().astype(np.uint8)
                prediction_image = img_copy.copy().astype(np.uint8)

                # Separate the x and y coordinates for both eyes and the mouth
                x_coord_right_eye = [sublist[0] for sublist in relative_source_right_eye_points]
                x_coord_left_eye = [sublist[0] for sublist in relative_source_left_eye_points]
                y_coord_left_eye = [sublist[1] for sublist in relative_source_left_eye_points]
                y_coord_right_eye = [sublist[1] for sublist in relative_source_right_eye_points]
                y_coord_mouth = [sublist[1] for sublist in relative_source_mouth_points]
                x_coord_mouth = [sublist[0] for sublist in relative_source_mouth_points]
                predicted_y_coordinates_eyes = y_coord_left_eye + y_coord_right_eye

                # Initialize the horizontal lines during the first 30 frames
                if first_thirty_frame:
                    first_thirty_frame_points_yellow.append(int(max(predicted_y_coordinates_eyes)))
                    first_thirty_frame_points_red.append(int(min(y_coord_mouth)))
                    first_thirty_frame_counter += 1
                    if 30 == first_thirty_frame_counter:
                        first_thirty_frame = False
                        set_lines = True

                    nodding_values.append(0)
                elif set_lines:
                    yellowHorizontal = np.mean(first_thirty_frame_points_yellow)
                    redHorizontal = np.mean(first_thirty_frame_points_red)
                    set_lines = False
                    yellowLineStart = (int(0), int(yellowHorizontal))
                    yellowLineEnd = (int(image.shape[1]), int(yellowHorizontal))
                    redLineStart = (int(0), int(redHorizontal))
                    redLineEnd = (int(image.shape[1]), int(redHorizontal))

                    nodding_values.append(0)

                    logger.debug("Initialized red and yellow horizontal lines for nodding detection.")
                # Wait for 5 frames to finalize nodding event
                if failed_to_detect_face and face_detected_again_counter < 5:
                    face_detected_again_counter += 1
                    nodding_values.append(1)
                else:
                    if not set_lines and not first_thirty_frame:
                        cv2.line(img_copy, yellowLineStart, yellowLineEnd, (0, 255, 255))
                        cv2.line(img_copy, redLineStart, redLineEnd, (0, 0, 255))

                        # Check if eyes are below the threshold or not
                        if min(predicted_y_coordinates_eyes) > yellowLineStart[1]:
                            if max(predicted_y_coordinates_eyes) > redLineStart[1]:
                                if prevState == 1:
                                    eventOngoing = True
                                numberOfFramesInRed += 1
                                prevState = 2
                                underRedLine = True
                            else:
                                prevState = 1
                                numberOfFramesInYellow += 1
                                underRedLine = False
                        else:
                            if eventOngoing:
                                eventOngoing = False
                                noOfNods += 1
                            prevState = 0
                            numberOfFramesInRed = 0
                            numberOfFramesInYellow = 0
                        if eventOngoing:
                            nodding_values.append(1)
                        else:
                            nodding_values.append(0)

                # Calculate the area of the eyes
                left_area = (
                        abs(int(min(x_coord_left_eye)) - int(max(x_coord_left_eye))) *
                        abs(int(min(y_coord_left_eye)) - int(max(y_coord_left_eye))))
                right_area = (abs(int(min(x_coord_right_eye)) - int(max(x_coord_right_eye))) *
                              abs(int(min(y_coord_right_eye)) - int(max(y_coord_right_eye))))
                # Crop the eye that is better represented in the frame
                if left_area > right_area:
                    prediction_image_eye = get_prediction_image(x_coord_left_eye,
                                                            y_coord_left_eye, prediction_image)
                    #logging.debug("Making prediction using left eye.")
                else:
                    prediction_image_eye = get_prediction_image(x_coord_right_eye,
                                                            y_coord_right_eye, prediction_image)
                    #logging.debug("Making prediction using right eye.")
                """for i in range(len(predicted_x_coordinates_eyes_left)):
                    cv2.circle(img_crop_resized, (int(predicted_x_coordinates_eyes_left[i]), int(predicted_y_coordinates_eyes_left[i])), 2, (0, 0, 255), -1)"""
                # Crop the region containing the mouth
                prediction_image_mouth = get_prediction_image(x_coord_mouth, y_coord_mouth, prediction_image)
                
                idx += 1
                # Collect ten images to perform batch prediction
                if len(prediction_batch_eye) < 10:
                    prediction_batch_eye.append(prediction_image_eye)
                # pred_array = [prediction_image]
                # pred_array = np.asarray(pred_array)
                # Perform batch prediction
                elif len(prediction_batch_eye) == 10:
                    prediction_batch_eye = np.asarray(prediction_batch_eye)
                    model_predict_start = time.time()
                    prediction_eye = blink_model.predict(prediction_batch_eye, verbose=0)
                    model_predict_end = time.time()
                    model_predict_time_array.append(model_predict_end - model_predict_start)
                    prediction_batch_eye = []
                    prediction_batch_eye.append(prediction_image_eye)
                    # Process the predictions made by the model, 0 for Blink, 1 for No Blink
                    for i in range(len(prediction_eye)):
                        if round(prediction_eye[i][0]) == 0:
                            prediction_blink_counter += 1
                        elif round(prediction_eye[i][0]) == 1 and (not(0 == i) and round(prediction_eye[i - 1][0])) == 1 and prediction_blink_counter != 0:
                            # Logging
                            if prediction_blink_counter > 1:
                                blinks_predicted.append(time.perf_counter_ns())
                                blink_durations_predicted.append(prediction_blink_counter)
                                """cv2.putText(img_copy, f"BLINK HAPPENED (PREDICTED)!", (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,(255, 255, 255), 2)"""
                                logging.debug("Blink happened (predicted), duration: %f", prediction_blink_counter)
                            prediction_blink_counter = 0
                        ear_prediction_values.append(round(prediction_eye[i][0], 2))
                # Collect 90 images to perform batch prediction
                if len(prediction_batch_mouth) < 90:
                    prediction_batch_mouth.append(prediction_image_mouth)
                # Perform batch prediction
                elif 90 == len(prediction_batch_mouth):
                    prediction_batch_mouth = np.asarray(prediction_batch_mouth)
                    prediction_mouth = yawn_model.predict(prediction_batch_mouth, verbose=0)
                    logging.debug(prediction_mouth)
                    prediction_batch_mouth = []
                    prediction_batch_mouth.append(prediction_image_mouth)
                    # Process the predictions made by the model, 0 for No Yawn, 1 for Yawn
                    for i in range(len(prediction_mouth)):
                        if round(prediction_mouth[i][0]) == 1:
                            prediction_yawn_counter += 1
                        elif round(prediction_mouth[i][0]) == 0 and (not(0 == i) and round(prediction_mouth[i - 1][0]) == 0) and prediction_yawn_counter != 0:
                            if prediction_yawn_counter > YAWN_THRESHOLD:
                                yawns_predicted.append(time.perf_counter_ns())
                                yawn_durations_predicted.append(prediction_yawn_counter)
                                logging.debug("Yawn happened (predicted), duration: %f", prediction_yawn_counter)
                                prediction_yawn_counter = 0
                        mouth_prediction_values.append(round(prediction_mouth[i][0],2))

                # Putting the number of events on the screen
                cv2.putText(img_copy, f"Number of blinks (calculated): {len(blink_durations_calculated)}", (0, 150), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
                cv2.putText(img_copy, f"Number of blinks (predicted): {len(blink_durations_predicted)}", (0, 180), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
                cv2.putText(img_copy, f"Number of yawns (calculated): {len(yawn_durations)}", (0, 210), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
                cv2.putText(img_copy, f"Number of yawns (predicted): {len(yawn_durations_predicted)}", (0, 240), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
                cv2.putText(img_copy, f"Number of nods: {noOfNods}", (0, 270), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)

                if numberOfFramesInRed > 20:
                    cv2.putText(img_copy, "WARNING NOD!", (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Check if eyes are closed for too long
                if eyes_closed_consecutively >= 150:
                    is_drowsy = True
                # Check for drowsiness based on the predictions
                else:
                    is_drowsy = check_drowsiness(blink_durations_calculated, blinks_calculated, yawn_durations, yawns,
                                                 numberOfFramesInRed)
                if is_drowsy:
                    playsound(r'soft-alert.wav', block=False)
                    cv2.putText(img_copy, f"DROWSY ALERT!", (0, 300), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 2)

                """cv2.imwrite(f"Analysed_images\\webcam\\mouth\\{idx}.png", prediction_image_mouth*255)
                cv2.imwrite(f"Analysed_images\\webcam\\eye\\{idx}.png", prediction_image_eye * 255)
                """

                #cv2.imwrite(result_folder + f'\\{idx}.png', img_copy)

                # cv2.imwrite(f"Analysed_images\\classifier\\face\\{idx}.png", img_copy)
                # Show the live webcam feed
                cv2.imshow('Video', img_copy)
                #cv2.imshow('eye', prediction_image_eye)
                #cv2.imshow('Mouth', prediction_image_mouth)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                   break
                output_video.write(img_copy)
                frame_end = time.time()
                # print("Time of iteration: ", frame_end - frame_start)



                time_array.append(frame_end - frame_start)

        else:
            print("Reading from video failed")
            break

end_time = time.time()
# Print statistics
elapsed_time = end_time - start_time
print("Failed because of no face detected: ", failed_no_face)
print("Failed because of small IoU: ", failed_bcuz_iou)
print("Successfully detected: ", successful)
print("Time elapsed: ", elapsed_time)
print("No. of analysed frames: ", idx)
avg = statistics.mean(time_array)
maximum = max(time_array)

cap.release()
output_video.release()
cv2.destroyAllWindows()
# Save data for later analysis
with open(result_folder + '\\mar_values.txt', 'w') as file:
    file.write(",".join(map(str, MAR_values)))

with open(result_folder + '\\mar_threshold_values.txt', 'w') as file:
    file.write(",".join(map(str, MAR_threshold_values)))

with open(result_folder + '\\ear_values.txt', 'w') as file:
    file.write(",".join(map(str, EAR_values)))

with open(result_folder + '\\ear_threshold_values.txt', 'w') as file:
    file.write(",".join(map(str, EAR_threshold_values)))

with open(result_folder + '\\nodding_values.txt', 'w') as file:
    file.write(",".join(map(str, nodding_values)))

with open(result_folder + '\\mouth_prediction_values.txt', 'w') as file:
    file.write(",".join(map(str, mouth_prediction_values)))

with open(result_folder + '\\ear_prediction_values.txt', 'w') as file:
    file.write(",".join(map(str, ear_prediction_values)))

print("AVG: ", avg)
print("MAX: ", maximum)
print("Mediapipe process AVG: ", statistics.mean(mp_process_time_array))
print("Mediapipe process MAX: ", max(mp_process_time_array))
print("Model predict process AVG: ", statistics.mean(model_predict_time_array))
print("Model predict process MAX: ", max(model_predict_time_array))
logging.debug("Time elapsed: %f", elapsed_time)