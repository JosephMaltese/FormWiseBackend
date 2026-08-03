from utils.stageEnum import Stage
import cv2 as cv
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
from utils.constants import frame_skip
from utils.helperFunctions import calculate_angle, eccentric_score
import numpy

def analyze_squat(output, capture):
        attempted_rep_counter = 0
        complete_rom_rep_counter = 0
        partial_rom_rep_counter = 0
        stage = Stage.NONE
        min_torso_angle_in_rep = None
        max_torso_angle_in_rep = None
        last_recorded_torso_angle = None
        last_recorded_toe_knee_angle = None
        fps = capture.get(cv.CAP_PROP_FPS)
        width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        frame_count = 0
        processed_frame_count = 0
        eccentric_durations = []
        eccentric_start_time = None
        min_and_max_rep_angles = []
        current_rep_foot_cheat = False
        foot_cheat_reps = 0
        toe_knee_aligned_reps = 0
        horizontal_thigh_reps = 0
        current_rep_horizontal_thigh = False
        rep_in_progress = False

        def classify_and_store_rep(min_angle, max_angle):
            nonlocal complete_rom_rep_counter, partial_rom_rep_counter
            if min_angle < 90.0 and max_angle > 160.0:
                complete_rom_rep_counter += 1
                print("Complete rep")
            else:
                partial_rom_rep_counter += 1
                print("Partial rep")
            min_and_max_rep_angles.append((min_angle, max_angle))

        def reset_rep_tracking():
            nonlocal min_torso_angle_in_rep, max_torso_angle_in_rep
            nonlocal current_rep_foot_cheat, current_rep_horizontal_thigh, rep_in_progress
            min_torso_angle_in_rep = None
            max_torso_angle_in_rep = None
            current_rep_foot_cheat = False
            current_rep_horizontal_thigh = False
            rep_in_progress = False

        def finalize_rep(torso_angle=None):
            nonlocal attempted_rep_counter, min_torso_angle_in_rep, max_torso_angle_in_rep

            if not rep_in_progress or min_torso_angle_in_rep is None or max_torso_angle_in_rep is None:
                return False

            if torso_angle is not None:
                min_torso_angle_in_rep = min(min_torso_angle_in_rep, torso_angle)
                max_torso_angle_in_rep = max(max_torso_angle_in_rep, torso_angle)

            if abs(max_torso_angle_in_rep - min_torso_angle_in_rep) < 20.0:
                return False

            attempted_rep_counter += 1
            classify_and_store_rep(min_torso_angle_in_rep, max_torso_angle_in_rep)
            reset_rep_tracking()
            return True

        with mp_pose.Pose(
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as pose:
            while capture.isOpened():
                isTrue, frame = capture.read()
                if not isTrue:
                    break

                frame_count += 1
                if frame_count % frame_skip == 0:
                    continue

                processed_frame_count += 1

                frame.flags.writeable = False
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                results = pose.process(frame)

                frame.flags.writeable = True
                frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

                current_time = processed_frame_count / fps

                if results.pose_landmarks is not None:
                    try:
                        landmarks = results.pose_landmarks.landmark
                        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
                        left_toe = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

                        vertical = [left_knee[0], left_knee[1] + 0.2]

                        if left_toe[0] > left_heel[0]:
                            foot_horizontal = [left_toe[0] + 0.2, left_toe[1]]
                            thigh_horizontal = [left_knee[0] + 0.2, left_knee[1]]
                        else:
                            foot_horizontal = [left_toe[0] - 0.2, left_toe[1]]
                            thigh_horizontal = [left_knee[0] - 0.2, left_knee[1]]

                        toe_knee_angle = calculate_angle(left_toe, left_knee, vertical)
                        foot_floor_angle = calculate_angle(left_toe, left_heel, foot_horizontal)
                        thigh_angle = calculate_angle(thigh_horizontal, left_knee, left_hip)
                        torso_angle = calculate_angle(left_knee, left_hip, left_shoulder)

                        cv.putText(frame, f"TK:{int(toe_knee_angle)}",
                                        tuple(numpy.multiply(left_knee, [width, height]).astype(int)),
                                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
                        cv.putText(frame, f"FF:{int(foot_floor_angle)}",
                                        tuple(numpy.multiply(left_toe, [width, height]).astype(int)),
                                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
                        cv.putText(frame, f"TH:{int(thigh_angle)}",
                                        tuple(numpy.multiply(left_hip, [width, height]).astype(int)),
                                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
                        cv.putText(frame, f"TR:{int(torso_angle)}",
                                        tuple(numpy.multiply(left_shoulder, [width, height]).astype(int)),
                                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)

                        if last_recorded_torso_angle is not None:
                            if rep_in_progress and foot_floor_angle > 15.0 and not current_rep_foot_cheat:
                                current_rep_foot_cheat = True
                                foot_cheat_reps += 1

                            # Eccentric: moving down (torso angle decreasing)
                            if (torso_angle - last_recorded_torso_angle <= -5.0):
                                if stage == Stage.CONCENTRIC and rep_in_progress:
                                    if finalize_rep(torso_angle):
                                        stage = Stage.NONE
                                    else:
                                        reset_rep_tracking()

                                if stage == Stage.NONE or stage == Stage.CONCENTRIC:
                                    if stage == Stage.NONE:
                                        eccentric_start_time = current_time
                                    stage = Stage.ECCENTRIC

                                if stage == Stage.ECCENTRIC:
                                    last_recorded_toe_knee_angle = toe_knee_angle
                                    if 175.0 <= thigh_angle <= 185.0 and not current_rep_horizontal_thigh:
                                        current_rep_horizontal_thigh = True
                                        horizontal_thigh_reps += 1

                                last_recorded_torso_angle = torso_angle

                            # Concentric: moving up (torso angle increasing)
                            elif (torso_angle - last_recorded_torso_angle >= 5.0):
                                if stage == Stage.ECCENTRIC:
                                    stage = Stage.CONCENTRIC
                                    rep_in_progress = True
                                    current_rep_horizontal_thigh = False

                                    if last_recorded_toe_knee_angle is not None and last_recorded_toe_knee_angle <= 15.0:
                                        toe_knee_aligned_reps += 1

                                    min_torso_angle_in_rep = min(torso_angle, last_recorded_torso_angle)
                                    max_torso_angle_in_rep = torso_angle

                                    if eccentric_start_time is not None:
                                        eccentric_duration = current_time - eccentric_start_time
                                        eccentric_durations.append(eccentric_duration)
                                        print(f"Eccentric duration: {eccentric_duration:.2f}s")
                                        eccentric_start_time = None

                                elif stage == Stage.CONCENTRIC and rep_in_progress and torso_angle >= 160.0:
                                    if finalize_rep(torso_angle):
                                        stage = Stage.NONE

                                last_recorded_torso_angle = torso_angle
                        else:
                            last_recorded_torso_angle = torso_angle

                        if min_torso_angle_in_rep is not None:
                            min_torso_angle_in_rep = min(min_torso_angle_in_rep, torso_angle)
                            max_torso_angle_in_rep = max(max_torso_angle_in_rep, torso_angle)
                    except Exception as e:
                        print(f"Squat analysis error: {e}")

                cv.rectangle(frame, (0, 0), (500, 73), (245, 117, 16), -1)
                cv.putText(frame, 'REPS', (15, 12),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(frame, str(attempted_rep_counter),
                            (10, 60),
                            cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)
                cv.putText(frame, 'STAGE', (95, 12),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(frame, stage.name,
                            (90, 60),
                            cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

                if results.pose_landmarks is not None:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                output.write(frame)

        # Count the final rep if the set ended while standing (no subsequent eccentric)
        if rep_in_progress:
            finalize_rep()

        print("Eccentric durations:", eccentric_durations)
        print("Total rep count:", attempted_rep_counter)
        print("Complete ROM rep count:", complete_rom_rep_counter)
        print("Partial ROM rep count:", partial_rom_rep_counter)
        print("Foot cheat reps:", foot_cheat_reps)
        print("Toe knee aligned reps:", toe_knee_aligned_reps)
        print("Horizontal thigh aligned reps:", horizontal_thigh_reps)

        total_score = calculate_squat_score(
            attempted_rep_counter,
            complete_rom_rep_counter,
            partial_rom_rep_counter,
            eccentric_durations,
            foot_cheat_reps,
            toe_knee_aligned_reps,
            horizontal_thigh_reps,
        )

        return (total_score,
            attempted_rep_counter,
            complete_rom_rep_counter,
            partial_rom_rep_counter,
            eccentric_durations,
            min_and_max_rep_angles,
            foot_cheat_reps,
            toe_knee_aligned_reps,
            horizontal_thigh_reps)

def calculate_squat_score(attempted_rep_counter, complete_rom_rep_counter, partial_rom_rep_counter, eccentric_durations, foot_cheat_reps, toe_knee_aligned_reps, horizontal_thigh_reps)->int:
    if attempted_rep_counter == 0:
        return 0

    total_score = 0

    # Award 10 points for completing at least one valid repetition
    total_score += 10

    # ROM: 2 pts full, 1 pt partial (max 25)
    possible_rom_points = attempted_rep_counter * 2
    achieved_rom_points = (complete_rom_rep_counter * 2) + partial_rom_rep_counter
    total_score += (achieved_rom_points / possible_rom_points) * 25

    # Foot stability: penalize heel lift (max 20)
    stable_reps = attempted_rep_counter - foot_cheat_reps
    total_score += (stable_reps / attempted_rep_counter) * 20

    # Knee-over-toe alignment at bottom (max 15)
    total_score += (toe_knee_aligned_reps / attempted_rep_counter) * 15

    # Thigh parallel to floor at bottom (max 10)
    total_score += (horizontal_thigh_reps / attempted_rep_counter) * 10

    # Eccentric tempo (max 20)
    possible_duration_points = attempted_rep_counter * 10
    duration_points = sum(eccentric_score(duration) for duration in eccentric_durations)
    if possible_duration_points > 0:
        total_score += (duration_points / possible_duration_points) * 20

    return round(total_score)