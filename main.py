from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query, Form
from pydantic import BaseModel
from typing import Dict, Optional
import time
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

app = FastAPI()

# 작업 상태 저장을 위한 임시 딕셔너리
task_status: Dict[str, str] = {}

# S3 설정
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "test-fastapi-bucket")
S3_REGION_NAME = os.getenv("S3_REGION_NAME", "ap-northeast-2")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=S3_REGION_NAME
)

# S3에 파일을 업로드하는 함수
def upload_to_s3(file, filename, content_type):
    try:
        s3_client.upload_fileobj(
            file,
            S3_BUCKET_NAME,
            filename,
            ExtraArgs={"ContentType": content_type}
        )
        return f"https://{S3_BUCKET_NAME}.s3.{S3_REGION_NAME}.amazonaws.com/{filename}"
    except NoCredentialsError:
        raise Exception("AWS credentials not found.")

# S3에서 파일 다운로드 함수
def download_from_s3(filename, download_path):
    try:
        # 파일 존재 여부 확인
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=filename)

        # S3에서 파일 다운로드
        s3_client.download_file(
            Bucket=S3_BUCKET_NAME,
            Key=filename,
            Filename=download_path
        )
        return f"파일이 '{download_path}'로 다운로드되었습니다."
    except NoCredentialsError:
        return "AWS 자격 증명이 누락되었습니다."
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return "파일이 S3에 존재하지 않습니다."
        else:
            return f"파일 다운로드 실패: {str(e)}"

# 영상 처리를 비동기로 수행하는 함수
def process_video(s3_url: str, task_id: str):
    time.sleep(5)
    task_status[task_id] = "완료"  # 작업 상태 업데이트

# 영상 파일 업로드 및 처리 API
@app.post("/upload/")
async def upload_video(
    file: UploadFile = File(...), 
    dramaTitle: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    task_id = str(int(time.time()))  # 간단한 작업 ID 생성
    task_status[task_id] = "처리 중"

    # S3에 파일 업로드
    filename = f"{task_id}_{file.filename}"
    s3_url = upload_to_s3(file.file, filename, file.content_type)

    # 비동기로 영상 처리 작업 수행
    background_tasks.add_task(process_video, s3_url, task_id)

    return {
        "task_id": task_id,
        "status": "업로드 및 처리 중",
        "s3_url": s3_url,
        "dramaTitle": dramaTitle  # dramaTitle 반환
    }

# 작업 상태 확인 API
@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    status = task_status.get(task_id, "작업 ID가 존재하지 않음")
    return {"task_id": task_id, "status": status}

# 파일 다운로드 API
@app.get("/download/{filename}")
async def download_file(filename: str, download_path: Optional[str] = Query(None, description="저장할 로컬 경로")):
    if download_path is None:
        raise HTTPException(status_code=400, detail="다운로드 경로가 필요합니다.")

    # 파일 다운로드
    result = download_from_s3(filename, download_path)
    
    if "다운로드되었습니다" in result:
        return {"message": result, "local_path": download_path}
    else:
        raise HTTPException(status_code=400, detail=result)
