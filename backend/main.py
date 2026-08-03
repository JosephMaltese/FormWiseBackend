# uvicorn main:app --reload
# source venv/bin/activate
from fastapi import FastAPI, UploadFile, File, Form
import shutil
import cv2 as cv
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
import json
import squat
import bench
import bicepCurl
from utils.constants import frame_skip

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, PUT, DELETE etc.
    allow_headers=["*"],  # allow all headers
    expose_headers=[
        "total_score", 
        "gpt_analysis", 
        "rep_count", 
        "complete_rom_rep_count", 
        "partial_rom_rep_count", 
        "cheat_rep_count", 
        "eccentric_durations", 
        "min_and_max_rep_angles"
        ],
)

@app.post("/analyze-video/")
async def analyze_video(file: UploadFile = File(...), exercise: str = Form(...)):
    video = file.file
    with open("input.mp4", "wb") as buffer:
        shutil.copyfileobj(video, buffer)
    
    capture = cv.VideoCapture("input.mp4")
    fourcc = cv.VideoWriter_fourcc(*"avc1")
    fps = int(capture.get(cv.CAP_PROP_FPS))
    output_fps = fps * (frame_skip - 1) / frame_skip
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    output = cv.VideoWriter("processed.mp4", fourcc, output_fps, (width, height))

    foot_cheat_reps = 0
    toe_knee_aligned_reps = 0
    horizontal_thigh_reps = 0
    if exercise == "BICEP CURL":
        (score, rep_count,
         complete_rom_rep_count,
         partial_rom_rep_count,
         cheat_rep_count,
         eccentric_durations, 
         min_and_max_rep_angles) = bicepCurl.analyze_bicep_curl(output, capture)
    elif exercise == "BENCH PRESS":
        (score,
         rep_count,
         complete_rom_rep_count,
         partial_rom_rep_count, 
         cheat_rep_count, eccentric_durations,
         min_and_max_rep_angles) = bench.analyze_bench_press(output, capture)
    elif exercise == "SQUAT":
        (score,
         rep_count,
         complete_rom_rep_count,
         partial_rom_rep_count,
         eccentric_durations,
         min_and_max_rep_angles,
         foot_cheat_reps,
         toe_knee_aligned_reps,
         horizontal_thigh_reps) = squat.analyze_squat(output, capture)
        cheat_rep_count = 0
    else:
        score = 0
        rep_count = 0
        complete_rom_rep_count = 0
        partial_rom_rep_count = 0
        cheat_rep_count = 0
        eccentric_durations = []
        min_and_max_rep_angles = []

    capture.release()
    output.release()
    
    return FileResponse(
        "processed.mp4",
        media_type="video/mp4",
        filename="processed.mp4",
        headers={
            "total_score": str(score),
            "rep_count": str(rep_count),
            "complete_rom_rep_count": str(complete_rom_rep_count),
            "partial_rom_rep_count": str(partial_rom_rep_count),
            "cheat_rep_count": str(cheat_rep_count),
            "eccentric_durations": json.dumps(eccentric_durations),
            "min_and_max_rep_angles": json.dumps(min_and_max_rep_angles),
            "foot_cheat_reps": json.dumps(foot_cheat_reps),
            "toe_knee_aligned_reps": json.dumps(toe_knee_aligned_reps),
            "horizontal_thigh_reps": json.dumps(horizontal_thigh_reps),
            }
    )

@app.get("/ai-summary/")
async def ai_summary(exercise: str = Form(...), rep_count: str = Form(...), complete_rom_rep_count: str = Form(...), partial_rom_rep_count: str = Form(...), cheat_rep_count: str = Form(...), eccentric_durations: list[int] = Form(...), min_and_max_rep_angles: list[int] = Form(...)) -> str:
    developer_prompt = "You are a helpful personal fitness assistant, providing feedback to a user regarding their exercise form. They will tell you which exercise they performed, and will provide some details about how they performed it (ex. relevant joint angles, eccentric rep duration, range of motion, etc.), and you will respond with a paragraph of constructive feedback, noting both the highlights and the drawbacks (if applicable) of their set. Do not judge them based on the number of repetitions they performed, as they only have 15 seconds to perform their set, so the purpose of this is to judge only their form while performing the set. They will tell you the total number of repetitons they performed, the number of those reps which were performed with complete range of motion (ROM), the number of those reps which were performed with partial ROM, the number of those which were performed with incomplete ROM, the number of those reps which were cheat reps, as well as a list of the durations of the eccentric motion of their reps, and a list containing a tuple for each rep, where the first element in the tuple is the minimum angle in the rep and the second element is the maximum angle in the rep. For clarity, a rep counts as any significant up + down movement. We use the angle between the elbow, shoulder, and wrist when analyzing the form. A rep with complete ROM is one where the minimum angle is <= 40 degrees and the maximum angle is >= 160 degrees. A partial ROM rep is one where one of the those two conditions don't hold. An incomplete rep is one where neither of those conditions are met. When analyzing cheat reps, we look at the angle between the hip, shoulder, and the vertical. If this angle deviates more than 6 degrees during the rep, we count it as a cheat rep (meaning they either used momentum or used poor posture to get their arm up). The eccentric durations are the amount of time it takes for the bicep to go from the contracted position to an elongated position. If the eccentric duration for a rep is less than 0.5s, we assign it a score of zero. Otherwise, we use a Gaussian curve centered at 3s with a width of 1s. Thus, reps ideally have an eccentric duration of 2-4s. Do not get too technical (for example, saying the exact angle their arm should bend at), but instead explain it to them in an understandable manner as if you were a personal trainer (ex. 'try to squeeze the bicep at the top and go slower on your eccentric motion'). Do not include any filler text/phrases (such as saying 'absolutely!'). Just provide your analysis."
    user_prompt = f"Analyze my {exercise} form. I completed {rep_count} reps in total. {complete_rom_rep_count} of these reps were performed with complete ROM. {partial_rom_rep_count} of these reps were performed with partial ROM. {rep_count - complete_rom_rep_count - partial_rom_rep_count} of these reps were performed with incomplete ROM. {cheat_rep_count} of the reps were cheat reps. My eccentric durations were {eccentric_durations}. My minimum and maximum angles for each rep were {min_and_max_rep_angles}."

    gptResponse = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "developer", "content": developer_prompt},
        {"role": "user", "content": user_prompt}
    ]
    )

    gpt_analysis = gptResponse.choices[0].message.content
    #gpt_analysis = "great work!" # Temporary value
    encoded_analysis = base64.b64encode(gpt_analysis.encode('utf-8')).decode('ascii')
    
    return encoded_analysis