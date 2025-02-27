from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from typing import Dict, List, Any
import time
from botocore.exceptions import NoCredentialsError
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from io import BytesIO
from face_detection_and_clustering import face_detection_and_clustering
from emotion_detection import emotion_detection
import mimetypes
from s3_client import s3_client
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from person_score import person_score
import re
from adjust_highlights import scene_detection
from drama_crawling import search_drama, get_drama
from clip_video_info import clip_and_save_highlights, insert_title_into_video
from title_generation import generate_highlight_title

TEMP_DIR = 'tmp'

load_dotenv()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.get("/search")
async def search_drama_api(drama_title: str):
    result = search_drama(drama_title)
    print(f"🔍 검색 결과: {result}")  # 실제 검색 결과 확인
    if result:
        return {"status": "success", "data": result}
    else:
        raise HTTPException(status_code=404, detail="드라마 정보를 찾을 수 없습니다.")

@app.get("/get_drama")
async def get_drama_api(drama_title: str):
    result = get_drama(drama_title)
    if result:
        return result
    else:
        raise HTTPException(status_code=404, detail="드라마 정보를 찾을 수 없습니다.")

# 인물(배우) 클래스 
class Actor:
    def __init__(self, name: str, imgSrc: str):
        self.name = name
        self.imgSrc = imgSrc

# 작업 상태 저장을 위한 임시 딕셔너리
task_status: List[Dict[str, Any]] = []

# 작업 ID로 task_status 리스트에서 해당 작업을 찾는 함수
def get_task_by_id(task_id: str):
    return next((task for task in task_status if task["task_id"] == task_id), None)

# task_status에 새로운 작업을 추가하거나 업데이트하는 함수
def add_or_update_task(task_id: str, status: str, additional_data: Dict[str, Any] = None):
    task = get_task_by_id(task_id)
    if task:
        # 작업이 이미 존재하면 상태만 업데이트
        task["status"] = status
        if additional_data:
            task.update(additional_data)
    else:
        # 작업이 존재하지 않으면 새로 추가
        new_task = {"task_id": task_id, "status": status}
        if additional_data:
            new_task.update(additional_data)
        task_status.append(new_task)

# S3 설정
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "test-fastapi-bucket")
S3_REGION_NAME = os.getenv("S3_REGION_NAME", "ap-northeast-2")


def delete_specified_files(task_id, folder_path=TEMP_DIR):
    try:
        if os.path.exists(folder_path):
            # 폴더 내 모든 파일 탐색
            for filename in os.listdir(folder_path):
                # 파일 이름이 v{task_id}_frame... 형식인 파일 찾기
                if filename.startswith(f"v{task_id}"):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)  # 파일 삭제
                        # print(f"파일 {filename}이 삭제되었습니다.")
            print(f"task_id: {task_id}와 일치하는 모든 파일 삭제 완료.")
        else:
            print(f"폴더 {folder_path}가 존재하지 않습니다.")
    except Exception as e:
        print(f"파일 삭제 중 오류 발생: {e}")

@app.get("/download_shorts")
def download_short(file_name: str):
    print("file name: " + file_name)
    try:
        # S3에서 파일 가져오기
        s3_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_name)
        video_file = s3_object['Body'].read()  # 파일 내용 가져오기
        file_size = len(video_file)  # 파일 크기 계산
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"Video {file_name} not found in S3 bucket.")

    headers = {
        "Content-Length": str(file_size),
    }
    # 여러 개의 동영상 파일을 ZIP 등으로 묶어서 보내는 방법
    # 이 예시에서는 각 파일을 별도로 스트리밍으로 전송
    return StreamingResponse(BytesIO(video_file), media_type="video/mp4", headers=headers)

def upload_part(file, filename, upload_id, part_number, part_size):
    # 해당 파트 데이터 읽기
    part_data = file.read(part_size)

    # 파트 업로드
    part_response = s3_client.upload_part(
        Bucket=S3_BUCKET_NAME,
        Key=filename,
        UploadId=upload_id,
        PartNumber=part_number,
        Body=part_data
    )
    return part_number, part_response['ETag']

def upload_to_s3(file, filename, content_type):
    try:
        # 멀티파트 업로드 시작
        create_multipart_upload_response = s3_client.create_multipart_upload(
            Bucket=S3_BUCKET_NAME,
            Key=filename,
            ContentType=content_type
        )
        upload_id = create_multipart_upload_response['UploadId']

        # 업로드할 파트 수 계산
        file_size = os.path.getsize(file.name)  # 파일의 크기
        part_size = 5 * 1024 * 1024  # 5MB 단위로 분할 (최소 파트 크기)
        num_parts = (file_size // part_size) + \
            (1 if file_size % part_size > 0 else 0)

        # 파일 포인터를 처음으로 되돌리기 위해 다시 설정
        file.seek(0)

        # 파트 업로드할 스레드 풀 설정
        with ThreadPoolExecutor() as executor:
            futures = []
            part_info = []

            # 각 파트를 비동기적으로 업로드
            for part_number in range(1, num_parts + 1):
                futures.append(executor.submit(upload_part, file, filename, upload_id, part_number, part_size))

            # 모든 파트가 완료될 때까지 기다리고 결과 처리
            for future in as_completed(futures):
                part_number, etag = future.result()
                part_info.append({
                    'PartNumber': part_number,
                    'ETag': etag
                })
        part_info.sort(key=lambda x: x['PartNumber'])
        # 업로드 완료
        s3_client.complete_multipart_upload(
            Bucket=S3_BUCKET_NAME,
            Key=filename,
            UploadId=upload_id,
            MultipartUpload={'Parts': part_info}
        )

        # 파일 URL 반환
        return f"https://{S3_BUCKET_NAME}.s3.{S3_REGION_NAME}.amazonaws.com/{filename}"

    except NoCredentialsError:
        raise Exception("AWS credentials not found.")
    except Exception as e:
        # 실패 시 업로드 중지하고 실패한 부분 삭제
        s3_client.abort_multipart_upload(
            Bucket=S3_BUCKET_NAME,
            Key=filename,
            UploadId=upload_id
        )
        raise e

# 영상 처리를 비동기로 수행하는 함수
def process_video(s3_url: str, task_id: str):
    try:
        print(f"Processing video for task {task_id}...")
        # 감정 분석 수행
        result = emotion_detection(s3_url, task_id, emotion_threshold=10)
        # intervals, count = result

        # 감정 분석 결과 저장
        add_or_update_task(task_id, "감정 분석 완료", {
            "highlights": result[0],
            "highlight_count": result[1]
        })
        print(f"Emotion detection completed for task {task_id}: {result[1]} highlights")

    except Exception as e:
        add_or_update_task(task_id, "에러 발생", {"error": str(e)})
        print(f"Error processing video {task_id}: {e}")


# 감정 분석 결과를 조회하는 API 추가
@app.get("/tasks/{task_id}/emotion_highlights")
def get_emotion_highlights(task_id: str):
    task_result = get_task_by_id(task_id)
    if not task_result:
        raise HTTPException(status_code=404, detail="작업 ID를 찾을 수 없습니다.")
    
    if task_result.get("status") == "에러 발생":
        return {
            "status": "error",
            "message": "감정 분석 중 에러가 발생했습니다.",
            "error": task_result.get("error")
        }

    return {
        "status": task_result.get("status"),
        "highlights": task_result.get("highlights", []),
        "highlight_count": task_result.get("highlight_count", 0)
    }


def parse_s3_url(s3_url: str):
    regex = r"https://([^/]+)\.s3\.[^/]+\.amazonaws\.com/(.+)"
    match = re.match(regex, s3_url)
    if not match:
        raise ValueError(f"Invalid S3 URL: {s3_url}")
    return match.group(1), match.group(2)

def get_video_from_s3(s3_url):
    TEMP_DIR = "tmp"
    bucket_name, object_key = parse_s3_url(s3_url)
    local_path = os.path.join(TEMP_DIR, object_key.split('/')[-1])  # 임시 파일 경로 설정
    print(f"Downloading video from S3 to {local_path}")

    try:
        s3_client.download_file(bucket_name, object_key, local_path)
        print(f"Downloaded {local_path}")
    except NoCredentialsError:
        raise Exception("AWS credentials not available.")
    except Exception as e:
        raise Exception(f"Error downloading from S3: {e}")

# 업로드 후 감정 분석 자동 수행
@app.post("/upload")
async def upload_video(file: UploadFile = File(...), dramaTitle: str = Form(...), background_tasks: BackgroundTasks = None):
    task_id = str(int(time.time()))  # 간단한 작업 ID 생성
    add_or_update_task(task_id, "업로드 중")

    # S3에 파일 업로드
    filename = f"{task_id}_{file.filename}"
    s3_url = upload_to_s3(file.file, filename, file.content_type)
    print(f"성공적으로 업로드 완료 : {s3_url}")

    # 다운로드 to tmp 폴더(한 번에 비디오 처리 위함)
    get_video_from_s3(s3_url)

    # 비동기로 감정 분석 수행
    background_tasks.add_task(process_video, s3_url, task_id)

    return JSONResponse(content={
        "task_id": task_id,
        "status": "업로드 완료 및 감정 분석 진행 중",
        "s3_url": s3_url,
        "dramaTitle": dramaTitle
    })

@app.get("/person/dc")
def detect_and_cluster(s3_url: str, task_id: str):
    # 인물 감지 및 클러스터링
    print(f"------------------{task_id} 작업------------------")
    print(f"------------------인물 감지 및 클러스터링 시작: {s3_url}------------------")
    representative_images = face_detection_and_clustering(s3_url, task_id)  # Face Detection and Clustering
    if representative_images is None:
        return JSONResponse(content={"message": "클러스터링된 인물이 존재하지 않습니다."})
    image_urls = [upload_to_s3(open(image_path, 'rb'), os.path.basename(image_path), mimetypes.guess_type(image_path)[0] or 'application/octet-stream') for image_path in representative_images]
    delete_specified_files(task_id, TEMP_DIR)
    print(
        f"------------------인물 감지 및 클러스터링 완료 -> {image_urls}------------------")
    add_or_update_task(task_id, "인물 감지 및 클러스터링 완료", {"representative_images": image_urls})
    return JSONResponse(content={"message": "인물 감지와 클러스터링이 완료되었습니다.", "image_urls": image_urls})
    
@app.post("/api/videos/{video_id}/actors/select")
async def select_actors(video_id: str, request: Request):
    # 요청 본문을 JSON 형태로 출력
    body = await request.json()  # 요청 본문을 JSON으로 파싱
    print(f"요청 본문: {body}")  # 요청 본문 출력

    # 데이터가 예상대로 도달했는지 확인
    if "users" not in body:
        print("users 필드가 없습니다.")  # users 필드가 없으면 알림
        return {"message": "users 필드가 존재하지 않습니다.", "status": "error"}
    elif body['users'] == []:
        return {"message": "선택된 사용자가 없습니다.", "status": "error"}
    else:
        print("users 필드가 존재합니다.")

    # 정상적인 데이터 처리
    users = body.get("users", [])
    task_id = body.get("task_id")
    s3_url = body.get("s3_url")
    selected_actors = [Actor(user['name'], user['imgSrc']) for user in users]

    task = get_task_by_id(task_id)
    task["selected"] = selected_actors

    # task_status에 highlights와 highlight_count가 준비될 때까지 기다림
    while "highlights" not in task:
        print(f"Task {task_id}에서 감정 분석 결과가 준비될 때까지 기다리는 중...")
        await asyncio.sleep(1)  # highlights가 준비되기까지 1초 간격으로 대기

    # person_score 높은 순서 -> 낮은 순서로 정렬된 (내림차순) 하이라이트 리스트 가지고 오기
    print(f"--------------- task status : {task} ---------------")
    # print(f"--------------- task['highlights'], task['highlight_count'] : {task['highlights'], task['highlight_count']} ---------------")
    if (len(task['highlights']) == 0):
        return {"message": "추출된 하이라이트가 없습니다. 더 긴 영상의 비디오를 업로드해주세요.", "video_id": video_id, "data": users, "status": "no_highlight"}
    elif (len(task['highlights']) == 1):
        sorted_highlights = task['highlights']
    else:
        sorted_highlights = person_score(s3_url, task["highlights"], selected_actors)
    print(f"person scoring 기반으로 정렬된 하이라이트 : {sorted_highlights}")

    return {"message": "사용자 선택 완료", "video_id": video_id, "data": users, "status": "success", "sorted_highlights": sorted_highlights}

class HighlightRequest(BaseModel):
    s3_url: str
    task_id: str
    highlights: List[List[float]]

@app.post("/highlight/adjust")
async def detect_scenes(request: HighlightRequest):
    s3_url = request.s3_url
    task_id = request.task_id
    highlights = request.highlights

    print(f"------------------{task_id} 작업------------------")
    print(f"------------------하이라이트 조정 시작: {s3_url}, {highlights}------------------")

    _, object_key = parse_s3_url(s3_url)
    filename = object_key.split('/')[-1]
    base_name = filename.split('.')[0]
    local_path = os.path.join(TEMP_DIR, f"{base_name}.mov")  # .mov 확장자 사용

    adjusted_highlights = scene_detection(local_path, highlights, s3_url)  # scene_detection 함수 실행

    add_or_update_task(task_id, "하이라이트 조정 완료", {
        "adjusted_highlights": adjusted_highlights
    })

    print(f"------------------하이라이트 조정 완료 -> {adjusted_highlights}------------------")
    return JSONResponse(content={"message": "하이라이트 조정이 완료되었습니다.", "adjusted_highlights": adjusted_highlights})

class SaveClipRequest(BaseModel):
    s3_url: str
    task_id: str
    drama_title: str
    adjusted_highlights: List[List[float]]
    
@app.post("/highlights/save")
async def save_highlight_with_info(request: SaveClipRequest):
    s3_url = request.s3_url
    adjusted_highlights = request.adjusted_highlights
    task_id = request.task_id
    drama_title = request.drama_title

    print(f"------------------{task_id} 작업------------------")
    print(f"------------------쇼츠 제작 시작: {s3_url}, {adjusted_highlights}------------------")

    _, object_key = parse_s3_url(s3_url)
    filename = object_key.split('/')[-1]
    base_name = filename.split('.')[0]
    local_path = os.path.join(TEMP_DIR, f"{base_name}.mov")

    s3_url_list = clip_and_save_highlights(local_path, task_id, drama_title, adjusted_highlights, s3_url)

    print(f"------------------쇼츠 저장 완료 -> {s3_url_list}------------------")
    return JSONResponse(content={"message": "쇼츠 저장 완료", "s3_url_list": s3_url_list})

@app.post("/highlight/title")
def generate_short_title(org_title: str, file_name: str):
    titles = generate_highlight_title(org_title)
    print(f"titles: {titles}, file_name: {file_name}")
    return JSONResponse(content={"message": "title 추출 완료", "titles": titles, "file_name": file_name})


def extract_task_id_and_number(file_name):
    match = re.match(r"^(\d+)_highlight_with_info_(\d+)\.mp4$", file_name)
    if match:
        task_id = match.group(1)  # 처음의 숫자 (task_id)
        number = match.group(2)    # .mp4 앞의 숫자
        return task_id, number
    return None, None  # 형식이 맞지 않으면 None 반환

@app.post("/submit/title")
def insert_title_into_short(selected_title: str, file_name: str):
    task_id, idx = extract_task_id_and_number(file_name)

    TEMP_DIR = "tmp"
    selected_title = selected_title.replace('"', "")
    selected_title = selected_title.replace('-', "").strip()

    local_path = os.path.join(TEMP_DIR, file_name)
    s3_url = f"https://{S3_BUCKET_NAME}.s3.{S3_REGION_NAME}.amazonaws.com/{file_name}"
    s3_url_with_title = insert_title_into_video(local_path, task_id, selected_title, idx, s3_url)

    return JSONResponse(content={"message": "title 삽입 완료", "selected_title": selected_title, "s3_url_with_title": s3_url_with_title})