# 1단계: FFmpeg 설치
FROM jrottenberg/ffmpeg:7.1-ubuntu2404 as ffmpeg

# 2단계: 애플리케이션 이미지 구축
FROM python:3.9

# 작업 디렉토리 설정
WORKDIR /app

# FFmpeg 바이너리 복사
COPY --from=ffmpeg /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg

# 시스템 종속성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# FFmpeg 바이너리 다운로드 및 설치
RUN wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-i686-static.tar.xz -O /tmp/ffmpeg.tar.xz \
    && tar -xf /tmp/ffmpeg.tar.xz -C /tmp \
    && ls /tmp/ffmpeg*/bin/  # 바이너리 확인 (디버깅용) 
    && mv /tmp/ffmpeg*/bin/ffmpeg /usr/local/bin/ \
    && rm -rf /tmp/ffmpeg*

# FFmpeg 확인
RUN ffmpeg -version

# Python 종속성 설치
COPY ./requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# 애플리케이션 코드 복사
COPY app/ .

# 애플리케이션 실행 명령
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]