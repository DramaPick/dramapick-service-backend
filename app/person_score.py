from s3_client import s3_client
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError
import os
import re
import dlib
import cv2
import numpy as np

load_dotenv()

TEMP_DIR = 'tmp'

actor_embeddings = {}

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

face_detector = dlib.get_frontal_face_detector()  # 얼굴 탐지
sp = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')  # 얼굴 특징점 추출
face_encoder = dlib.face_recognition_model_v1('./dlib_face_recognition_resnet_model_v1.dat')  # 얼굴 임베딩 모델

def delete_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"파일 {file_path}가 삭제되었습니다.")
        else:
            print(f"파일 {file_path}가 존재하지 않습니다.")
    except Exception as e:
        print(f"파일 삭제 중 오류 발생: {e}")

def parse_s3_url(s3_url: str):
    regex = r"https://([^/]+)\.s3\.[^/]+\.amazonaws\.com/(.+)"
    match = re.match(regex, s3_url)
    if not match:
        raise ValueError(f"Invalid S3 URL: {s3_url}")
    return match.group(1), match.group(2)

# 얼굴 이미지에서 임베딩 벡터를 추출하는 함수
def encode_face(frame, shape=None):
    faces = face_detector(frame)
    if len(faces) == 0:
        return None  # 얼굴이 인식되지 않으면 None 반환
    # 첫 번째 얼굴만 처리
    face = faces[0]
    if shape is None:
        shape = sp(frame, face)  # 얼굴 특징점 추출
    face_embedding = face_encoder.compute_face_descriptor(frame, shape)  # 얼굴 임베딩 추출
    return np.array(face_embedding)


# 주어진 얼굴 임베딩과 배우의 임베딩을 비교하여 일치하는지 확인
def match_face(face_embedding, actor_name):
    if actor_name not in actor_embeddings:
        return False  # 배우의 임베딩이 없으면 False 반환

    actor_embedding = actor_embeddings[actor_name]
    # 두 얼굴 임베딩 사이의 유클리드 거리를 계산하여 비슷한지 비교
    distance = np.linalg.norm(face_embedding - actor_embedding)
    print(f"actor name : {actor_name}, distance : {distance}")
    if distance < 0.5:
        print("Same person.")
        return True
    else:
        print("Different person")
        return False


# 주어진 프레임에서 얼굴을 인식하고, 선택된 배우들과 일치하는지 확인
def detect_faces_in_frame(frame, selected_actors):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    matched_actors = set()  # 일치하는 배우들을 저장할 집합

    for face in faces:
        shape = sp(frame, face)
        face_embedding = encode_face(frame, shape)
        if face_embedding is None:
            continue
        for actor in selected_actors:
            if match_face(face_embedding, actor.name):
                matched_actors.add(actor.name)

    return matched_actors

def person_score(s3_url, intervals, selected_actors):
    bucket_name, object_key = parse_s3_url(s3_url)
    filename = object_key.split('/')[-1]
    base_name = filename.split('.')[0]  # 확장자 제외한 파일명
    local_path = os.path.join(TEMP_DIR, f"{base_name}.mov")  # 명시적으로 .mov 확장자 지정
    print(f"------------ PERSON SCORING local_path : {local_path} ------------")

    cap = cv2.VideoCapture(local_path)

    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {local_path}")
        return
    else:
        print("---------- 비디오 파일이 정상적으로 열렸습니다. ----------")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    # 얼굴 임베딩 등록
    img_src_list = []
    for actor in selected_actors:
        bucket_name, object_key = parse_s3_url(actor.imgSrc)
        local_path_ = os.path.join(TEMP_DIR, object_key.split('/')[-1])
        img_src_list.append(local_path_)
        try:
            s3_client.download_file(bucket_name, object_key, local_path_)
            print(f"이미지 {local_path_}로 S3에서 다운로드 완료")
        except NoCredentialsError:
            raise Exception("AWS credentials are not available.")
        except Exception as e:
            raise Exception(f"Error downloading image from S3: {e}")
        
        actor_face_image = cv2.imread(local_path_)
        face_embedding = encode_face(actor_face_image)
        if face_embedding is None:
            print(f"Error: 얼굴을 인식할 수 없습니다. {actor.name}의 얼굴 이미지를 다시 확인하세요.")
        else:
            actor_embeddings[actor.name] = face_embedding
            print(f"{actor.name}의 얼굴 임베딩이 저장되었습니다.")
    
    highlight_scores = []
    for start, end in intervals:
        print(f"Processing highlight interval: {start}-{end}")

        # 초 단위로 프레임 위치 계산
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)  # milliseconds 단위로 변환

        actor_appearance_count = {actor.name: 0 for actor in selected_actors}
        total_frames_in_highlight = 0

        current_time = start
        print(f"current_time: {current_time}, end: {end}")
        while current_time <= end:
            ret, frame = cap.read()
            print(f"ret : {ret}")
            if not ret:
                break

            print(f"---------- Processing time: {current_time} seconds ----------")

            matched_actors = detect_faces_in_frame(frame, selected_actors)
            for actor_name in matched_actors:
                actor_appearance_count[actor_name] += 1

            total_frames_in_highlight += 1

            current_time += 5
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)

        print(f"actor_appearance_count : {actor_appearance_count}")

        highlight_person_score = 0
        # 하이라이트 구간 내에 등장한 인물의 비율을 계산하고 person score 부여
        for actor in selected_actors:
            appearance_ratio = (actor_appearance_count[actor.name] / total_frames_in_highlight) if total_frames_in_highlight > 0 else 0
            highlight_person_score += appearance_ratio

        # 하이라이트 구간과 person score를 기록
        highlight_scores.append({
            'interval': [start, end],
            'score': highlight_person_score
        })

        print(f"highlight_scores: {highlight_scores}")

    # 작업이 끝난 후에 리소스 정리
    cap.release()
    # delete_file(local_path)
    for img_path in img_src_list:
        delete_file(img_path)

    # person score 기준 내림차순으로 정렬
    highlight_scores.sort(key=lambda x: x['score'], reverse=True)

    sorted_intervals = [(highlight['interval'][0], highlight['interval'][1]) for highlight in highlight_scores]
    print(f"Sorted highlight intervals by person score: {sorted_intervals}")
    return sorted_intervals