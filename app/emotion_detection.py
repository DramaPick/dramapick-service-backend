import cv2
import numpy as np
import os
import re
from deepface import DeepFace
from multiprocessing import Pool, cpu_count
from dotenv import load_dotenv

load_dotenv()

# --- 비디오에서 감정 분석 하이라이트 추출 ---
def extract_emotion_highlights(video_path, emotion_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 60  # 기본 60 FPS
    sampling_rate = (fps // 1)
    frames_to_process = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sampling_rate == 0:
            frames_to_process.append((frame_idx, frame, fps, emotion_threshold))

        frame_idx += 1

    cap.release()

    # 병렬 처리 실행
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_frame, frames_to_process)

    # None 값 제거
    highlights = [result for result in results if result is not None]
    return highlights


def process_frame(frame_info):
    frame_idx, frame, fps, emotion_threshold = frame_info

    frame_height, frame_width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    # 가장 큰 얼굴 선택
    largest_face = max(faces, key=lambda box: box[2] * box[3])
    x, y, w, h = largest_face
    face_area = w * h
    frame_area = frame_width * frame_height

    if face_area / frame_area < 0.05:
        return None

    # DeepFace로 감정 분석
    face = frame[y:y+h, x:x+w]
    try:
        analysis = DeepFace.analyze(face, actions=["emotion"], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = max(analysis, key=lambda result: max(result.get("emotion", {}).values()))

        emotion_scores = analysis.get("emotion", {})
        max_emotion = max(emotion_scores, key=emotion_scores.get)
        max_score = emotion_scores[max_emotion]
        
        if max_score > emotion_threshold:
            timestamp = frame_idx / fps
            print(f"Emotion Detected: {max_emotion} at {timestamp:.2f}s (Score: {max_score:.2f})")
            return {"timestamp": timestamp, "emotion": max_emotion, "score": max_score}

    except Exception as e:
        print(f"Error analyzing frame {frame_idx}: {e}")
        return None

# --- 감정 하이라이트 구간 병합 ---
def merge_emotional_intervals(highlights, min_duration=30, max_duration=60):
    timestamps = [highlight['timestamp'] for highlight in highlights]
    timestamps.sort()

    merged_intervals = []
    start = timestamps[0]
    end = start

    for i in range(1, len(timestamps)):
        if timestamps[i] - end <= 10:  # 5초 이내는 같은 구간으로 병합
            end = timestamps[i]
        else:
            duration = end - start
            if duration >= min_duration:
                merged_intervals.append([round(start, 2), 
                    round(min(end, start + max_duration), 2)])
            start = timestamps[i]
            end = start

    # 마지막 구간 추가
    if end - start >= min_duration:
        merged_intervals.append([start, min(end, start + max_duration)])

    return merged_intervals


# --- S3 URL 파싱 ---
def parse_s3_url(s3_url: str):
    regex = r"https://([^/]+)\.s3\.[^/]+\.amazonaws\.com/(.+)"
    match = re.match(regex, s3_url)
    if not match:
        raise ValueError(f"Invalid S3 URL: {s3_url}")
    return match.group(1), match.group(2)


# --- 파일 삭제 ---
def delete_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} deleted.")
        else:
            print(f"File {file_path} does not exist.")
    except Exception as e:
        print(f"Error deleting file: {e}")

# --- S3에서 비디오 다운로드 및 감정 분석 ---


def emotion_detection(s3_url, task_id, emotion_threshold=0.5):
    bucket_name, object_key = parse_s3_url(s3_url)
    filename = object_key.split('/')[-1]
    base_name = filename.split('.')[0]  # 확장자 제외한 파일명
    if "mov" in s3_url:
        local_path = os.path.join(
            "tmp", f"{base_name}.mov")  # 명시적으로 .mov 확장자 지정
    elif "mp4" in s3_url:
        local_path = os.path.join(
            "tmp", f"{base_name}.mp4")  # 명시적으로 .mp4 확장자 지정
    print(
        f"------------ EMOTION DETECTION local_path : {local_path} ------------")

    highlights = extract_emotion_highlights(local_path, emotion_threshold)

    # delete_file(local_path)

    if not highlights:
        print("No emotional highlights detected.")
        return [[], 0]

    # 하이라이트 구간 병합 및 반환
    merged_intervals = merge_emotional_intervals(highlights)
    return [merged_intervals, len(merged_intervals)]
