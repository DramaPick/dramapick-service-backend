# 1단계: FFmpeg 정적 빌드 다운로드 (첫 번째 FROM 사용)
FROM python:3.9 AS base

# 작업 디렉토리 설정
WORKDIR /app

# FFmpeg 정적 빌드 다운로드
RUN wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-i686-static.tar.xz -O /tmp/ffmpeg.tar.xz

# 다운로드한 FFmpeg 파일 압축 해제
RUN tar -xvf /tmp/ffmpeg.tar.xz -C /tmp

# 바이너리 파일을 /usr/local/bin/에 이동
RUN mv /tmp/ffmpeg-*-static/ffmpeg /usr/local/bin/
RUN mv /tmp/ffmpeg-*-static/ffprobe /usr/local/bin/

# 불필요한 파일 정리
RUN rm -rf /tmp/ffmpeg-*-static

# 2단계: 애플리케이션 이미지 구축
FROM python:3.9

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 종속성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    yasm \
    nasm \
    git \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

# FFmpeg 바이너리 복사 (첫 번째 단계에서 다운로드한 FFmpeg을 두 번째 단계로 복사)
COPY --from=base /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=base /usr/local/bin/ffprobe /usr/local/bin/ffprobe

# FFmpeg 버전 확인
RUN ffmpeg -version

# Python 종속성 설치
COPY ./requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# 애플리케이션 코드 복사
COPY app/ .

# 애플리케이션 실행 명령
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]