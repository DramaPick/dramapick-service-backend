# 1️⃣ Python 3.10 기반 이미지 사용
FROM python:3.10

# 2️⃣ 작업 디렉토리 설정
WORKDIR /app

# 3️⃣ 필수 패키지 설치 (dlib 종속 라이브러리 추가)
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libboost-all-dev \
    libopencv-dev

# 4️⃣ 프로젝트 파일 복사
COPY . /app/

# 5️⃣ requirements.txt를 설치하기 전에 pip 업그레이드
RUN pip install --upgrade pip

# 6️⃣ 패키지 설치 실행
RUN pip install --no-cache-dir -r /app/requirements.txt

# 7️⃣ FastAPI 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
