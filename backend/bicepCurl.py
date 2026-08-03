from utils.stageEnum import Stage
import cv2 as cv
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
from utils.constants import frame_skip
from utils.helperFunctions import calculate_angle, eccentric_score
import numpy as np

def analyze_bicep_curl(output, capture):
    attempted_rep_counter = 0
    complete_rom_rep_counter = 0
    partial_rom_rep_counter = 0
    cheat_rep_count = 0
    stage = Stage.NONE
    last_recorded_angle = None
    min_angle_in_rep = None
    max_angle_in_rep = None
    cheat_rep_detected = False
    torso_angle_at_rep_start = None
    starting_torso_angle_recorded = False
    fps = capture.get(cv.CAP_PROP_FPS)
    frame_count = 0
    eccentric_durations = []
    eccentric_start_time = None
    min_and_max_rep_angles = []

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while capture.isOpened():
            isTrue, frame = capture.read()
            if not isTrue:
                break

            frame_count += 1
            if frame_count % frame_skip == 0:
                continue

            frame.flags.writeable = False
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = pose.process(frame)

            frame.flags.writeable = True
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            current_time = frame_count / fps

            # Extract Landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                vertical = np.array([0, 1])

                angle = calculate_angle(shoulder, elbow, wrist)

                torso_angle = calculate_angle(shoulder, hip, np.array(hip) + vertical)

                # Visualize angles
                cv.putText(frame, str(angle),
                           tuple(np.multiply(elbow, [1620, 1080]).astype(int)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA
                           )
                cv.putText(frame, str(torso_angle),
                           tuple(np.multiply(hip, [1620, 1080]).astype(int)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA
                           )

                # Record starting torso angle when we're in the down position
                if angle >= 160.0 and not starting_torso_angle_recorded:
                    torso_angle_at_rep_start = torso_angle
                    starting_torso_angle_recorded = True

                # Check for cheating during the concentric phase (when going up)
                if stage == Stage.CONCENTRIC and angle <= 45.0 and starting_torso_angle_recorded:
                    torso_movement = abs(torso_angle - torso_angle_at_rep_start)
                    if torso_movement >= 6.0 and not cheat_rep_detected:
                        cheat_rep_detected = True
                        cheat_rep_count += 1
                        print(f"Cheat detected! Torso movement: {torso_movement:.1f}°")

                # rep counter logic
                if last_recorded_angle is not None:
                    if (angle - last_recorded_angle <= -5.0):
                        if (stage == Stage.NONE or stage == Stage.ECCENTRIC):
                            stage = Stage.CONCENTRIC

                            # Start tracking a new rep
                            min_angle_in_rep = angle
                            max_angle_in_rep = max(last_recorded_angle, angle)

                            if eccentric_start_time is not None:
                                eccentric_duration = current_time - eccentric_start_time
                                eccentric_durations.append(eccentric_duration)
                                print(f"Eccentric duration:", eccentric_duration)
                                eccentric_start_time = None

                        last_recorded_angle = angle
                    elif (angle - last_recorded_angle >= 5.0):
                        if stage == Stage.CONCENTRIC:
                            # First, ensure that the rep is valid and not just random movement (ex. walking)
                            if (abs(max_angle_in_rep - min_angle_in_rep) >= 20.0):
                                stage = Stage.ECCENTRIC
                                attempted_rep_counter += 1

                                eccentric_start_time = current_time

                                # Update min/max angles for this rep
                                if min_angle_in_rep is not None:
                                    min_angle_in_rep = min(min_angle_in_rep, angle)
                                    max_angle_in_rep = max(max_angle_in_rep, angle)

                                # Determine if this was a complete or partial rep
                                if min_angle_in_rep is not None and max_angle_in_rep is not None:
                                    if min_angle_in_rep < 40.0 and max_angle_in_rep > 160.0:
                                        complete_rom_rep_counter += 1
                                        print("Complete rep")
                                    else:
                                        partial_rom_rep_counter += 1
                                        print("Partial rep")

                                # Store min and max rep angles
                                min_and_max_rep_angles.append((min_angle_in_rep, max_angle_in_rep))

                                # Reset for next rep
                                min_angle_in_rep = None
                                max_angle_in_rep = None
                                cheat_rep_detected = False
                                starting_torso_angle_recorded = False
                        last_recorded_angle = angle
                else:
                    if angle > 140.0:
                        last_recorded_angle = angle

                # Update min/max angles during the rep
                if min_angle_in_rep is not None:
                    min_angle_in_rep = min(min_angle_in_rep, angle)
                    max_angle_in_rep = max(max_angle_in_rep, angle)
            except:
                pass

            # Render rep counter
            # Setup status box
            cv.rectangle(frame, (0, 0), (225, 73), (245, 117, 16), -1)

            # Rep data
            cv.putText(frame, 'REPS', (15, 12),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            cv.putText(frame, str(attempted_rep_counter),
                       (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

            # Stage data
            cv.putText(frame, 'STAGE', (65, 12),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            cv.putText(frame, stage.name,
                       (60, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

            # Cheat indicator
            if cheat_rep_detected:
                cv.putText(frame, 'CHEAT', (60, 60),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            output.write(frame)
    print("Eccentric durations:", eccentric_durations)
    print("Total rep count:", attempted_rep_counter)
    print("Complete ROM rep count:", complete_rom_rep_counter)
    print("Partial ROM rep count:", partial_rom_rep_counter)
    print("Cheat rep count:", cheat_rep_count)

    total_score = calculate_bicep_curl_score(attempted_rep_counter, complete_rom_rep_counter, partial_rom_rep_counter,
                                             cheat_rep_count, eccentric_durations)

    return (total_score,
            attempted_rep_counter,
            complete_rom_rep_counter,
            partial_rom_rep_counter,
            cheat_rep_count,
            eccentric_durations,
            min_and_max_rep_angles)

def calculate_bicep_curl_score(attempted_rep_counter, complete_rom_rep_counter, partial_rom_rep_counter, cheat_rep_count, eccentric_durations) -> int:
        # Initialize total form analysis score to 0
        total_score = 0

        # First, award the user 10 points for completing at least one valid repetition (up + down movement)
        if attempted_rep_counter >= 1:
            total_score += 10

        # Next, evaluate the ROM of the reps. Award 2 points for full ROM, 1 point for partial ROM (max. 20 score)
        possible_rom_points = attempted_rep_counter * 2
        acheived_rom_points = (complete_rom_rep_counter * 2) + (partial_rom_rep_counter)
        rom_score = (acheived_rom_points / possible_rom_points) * 30
        total_score += rom_score

        # Then, assess the stability of the user's form (max. 40 score)
        stable_reps = attempted_rep_counter - cheat_rep_count
        stability_score = (stable_reps / attempted_rep_counter) * 40
        total_score += stability_score

        # Finally, assess the duration of the user's eccentric motion (max. 20 score)
        possible_duration_points = attempted_rep_counter * 10
        duration_points = 0
        for duration in eccentric_durations:
            duration_points += eccentric_score(duration)
        duration_score = (duration_points / possible_duration_points) * 20
        total_score += duration_score

        return total_score