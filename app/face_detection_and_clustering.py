import numpy as np
import os
import cv2
from dotenv import load_dotenv
import dlib
import face_recognition
from imutils import face_utils
from numpy.linalg import norm
from sklearn.cluster import AgglomerativeClustering
from s3_client import s3_client
import re
import io
import multiprocessing

load_dotenv()

TEMP_DIR = 'tmp'

# S3 설정
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "test-fastapi-bucket")
S3_REGION_NAME = os.getenv("S3_REGION_NAME", "ap-northeast-2")

# dlib의 얼굴 감지기 및 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # 모델 파일 경로

def delete_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"파일 {file_path}가 삭제되었습니다.")
        else:
            print(f"파일 {file_path}가 존재하지 않습니다.")
    except Exception as e:
        print(f"파일 삭제 중 오류 발생: {e}")

# 얼굴에서 눈 감기 비율 계산
def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)  # EAR 계산
    return ear

# 화질 너무 좋지 않는 이미지 감지 함수
def getBlurScore(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()

# 얼굴에서 눈 감기 및 웃음 감지
def detect_smiling_and_eye(frame, faces, frame_number, frame_width, frame_height, task_id):
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
        face_width = x2 - x1
        face_height = y2 - y1

        if face_width < 500 or face_height < 500:
            continue

        margin_x = int(face_width * 0.1)
        margin_y = int(face_height * 0.1)
        new_x1 = max(x1 - margin_x, 0)
        new_y1 = max(y1 - margin_y, 0)
        new_x2 = min(x2 + margin_x, frame_width)
        new_y2 = min(y2 + margin_y, frame_height)

        face_image_ = frame[new_y1:new_y2, new_x1:new_x2]
        if getBlurScore(face_image_) < 5:
            continue  # 흐릿한 얼굴 건너뛰기

        gray_face_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks = predictor(gray_face_image, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        left_ear = eye_aspect_ratio(np.array(left_eye))
        right_ear = eye_aspect_ratio(np.array(right_eye))
        ear = (left_ear + right_ear) / 2.0

        lip_jaw_ratio = norm(landmarks[54] - landmarks[48]) / norm(landmarks[2] - landmarks[14])
        mouth_opening = norm(landmarks[57] - landmarks[51])
        mouth_nose = norm(landmarks[33] - landmarks[51])

        is_smiling = lip_jaw_ratio > 0.44 and mouth_opening / mouth_nose >= 1.05

        if ear < 0.2:
            if is_smiling:
                filename = os.path.join(TEMP_DIR, f"v{task_id}_frame_{frame_number}_face_{i + 1}_smiling_and_eye_closed.jpg")
                cv2.imwrite(filename, frame[new_y1:new_y2, new_x1:new_x2])
            else:
                continue
        else:
            filename = os.path.join(
                TEMP_DIR, f"v{task_id}_frame_{frame_number}_face_{i + 1}.jpg")
            cv2.imwrite(filename, frame[new_y1:new_y2, new_x1:new_x2])

# 얼굴 각도 계산 함수
def calculate_face_angle(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_landmarks_list = face_recognition.face_landmarks(image_rgb)

    if len(face_landmarks_list) > 0:
        face_landmarks = face_landmarks_list[0]
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        nose = face_landmarks['nose_bridge']

        eye_distance = np.linalg.norm(
            np.array(left_eye[0]) - np.array(right_eye[3]))
        nose_to_eyes_distance = np.linalg.norm(
            np.array(nose[0]) - np.array(left_eye[0]))
        angle = np.arctan(nose_to_eyes_distance / eye_distance)
        angle_deg = np.degrees(angle)
        return angle_deg
    else:
        return float('inf')


def parse_s3_url(s3_url: str):
    regex = r"https://([^/]+)\.s3\.[^/]+\.amazonaws\.com/(.+)"
    match = re.match(regex, s3_url)
    if not match:
        raise ValueError(f"Invalid S3 URL: {s3_url}")
    bucket_name = match.group(1)
    object_key = match.group(2)
    print(f"bucket_name : {bucket_name}, object_key: {object_key}")
    return bucket_name, object_key

def get_from_s3(s3_url: str):
    bucket_name, object_key = parse_s3_url(s3_url)
    # S3에서 파일 다운로드
    s3_object = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    # 파일 콘텐츠를 메모리 상에 저장
    video_data = s3_object['Body'].read()  # 이 데이터는 BytesIO 형태로 비디오 데이터를 반환합니다.
    # BytesIO로 변환하여 OpenCV나 다른 라이브러리에서 처리 가능하도록
    video_stream = io.BytesIO(video_data)
    return video_stream

# 얼굴 특징 추출 함수
def extract_face_features(image_path):
    if ".mp4" in image_path or ".mov" in image_path:
        return None, None
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image at {image_path}. Please check the file path or format.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(image_rgb)

    if len(encodings) > 0:
        return encodings[0], image_path
    else:
        return None, image_path

# 병렬화된 얼굴 특징 추출 함수
def extract_face_features_parallel(image_paths):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(extract_face_features, image_paths)

    # 유효한 결과만 필터링
    features = []
    filenames = []
    for encoding, filename in results:
        if encoding is None and filename is None:
            continue
        elif encoding is not None:
            features.append(encoding)
            filenames.append(filename)
        else:
            print(f"얼굴을 찾을 수 없음: {filename}")

    return features, filenames

# 병렬화된 얼굴 각도 계산 함수
def calculate_face_angle_parallel(image_paths):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        angles = pool.map(calculate_face_angle, image_paths)
    return angles


def face_detection_and_clustering(s3_url, task_id):
    _, object_key = parse_s3_url(s3_url)
    filename = object_key.split('/')[-1]
    base_name = filename.split('.')[0]  # 확장자 제외한 파일명
    if "mov" in s3_url:
        local_path = os.path.join("tmp", f"{base_name}.mov")  # 명시적으로 .mov 확장자 지정
    elif "mp4" in s3_url:
        local_path = os.path.join("tmp", f"{base_name}.mp4")  # 명시적으로 .mp4 확장자 지정 
    print(f"⏳ ------------ FACE DETECTION AND CLUSTERING -> local_path : {local_path} ------------")


    cap = cv2.VideoCapture(local_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = (int(fps) - 1) * 1
    frame_number = 0

    print(f"⏳ frame interval: {frame_interval}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        if frame_number % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            detect_smiling_and_eye(frame, faces, frame_number, frame_width, frame_height, task_id)

    cap.release()

    print("☎️ 프레임 저장 완료 ☎️")

    # 이미지 파일 목록
    image_files = os.listdir(TEMP_DIR)
    image_paths = []
    for file in image_files:
        if "mov" in os.path.join(TEMP_DIR, file):
            pass
        elif "mp4" in os.path.join(TEMP_DIR, file):
            pass
        else:
            image_paths.append(os.path.join(TEMP_DIR, file))
    
    features, filenames = extract_face_features_parallel(image_paths)

    print("☎️ 이미지 경로 mov 제외 리스트에 추가 완료 ☎️")
    if len(features) == 0:
        print("얼굴 인식된 이미지가 없습니다. 클러스터링을 수행할 수 없습니다.")
        return None

    # 코사인 유사도를 계산하여 군집화
    print("🪫 코사인 유사도 기반 군집화 학습 시작")
    clustering = AgglomerativeClustering(distance_threshold=0.05, n_clusters=None, metric='cosine', linkage='average')
    clustering.fit(features)

    print("🪫 대표 이미지 선정 준비")
    representative_images = []
    angles = calculate_face_angle_parallel(filenames)
    
    # 각 클러스터에 대해 대표 이미지 선정
    for cluster_id in np.unique(clustering.labels_):
        print(f"Cluster {cluster_id}:")
        cluster_indices = np.where(clustering.labels_ == cluster_id)[0]
        representative_image = None
        min_angle = float('inf')

        for idx in cluster_indices:
            # angle = calculate_face_angle(filenames[idx])
            angle = angles[idx]
            if angle < min_angle:
                min_angle = angle
                representative_image = filenames[idx]

        print(f"Representative Image for Cluster {cluster_id}: {representative_image}")
        representative_images.append(representative_image)

        if representative_image:
            output_image_path = os.path.join(TEMP_DIR, f"v{task_id}_cluster_{cluster_id}_representative.jpg")
            img = cv2.imread(representative_image)
            cv2.imwrite(output_image_path, img)

    print(f"클러스터링이 완료되었습니다. --> {representative_images}")
    return representative_images