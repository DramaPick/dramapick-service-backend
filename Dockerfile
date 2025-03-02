# 1단계: FFmpeg 설치
FROM jrottenberg/ffmpeg:7.1-ubuntu2404 as ffmpeg

# 2단계: 애플리케이션 이미지 구축
FROM python:3.9

# 작업 디렉토리 설정
WORKDIR /app

# FFmpeg 바이너리 복사
COPY --from=ffmpeg /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg

RUN add-apt-repository ppa:savoury1/ffmpeg4 && add-apt-repository ppa:savoury1/ffmpeg5

RUN apt-get update && apt-get upgrade && apt-get dist-upgrade && apt-get install ffmpeg

# FFmpeg 의존성 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libavdevice58 \
    libavfilter7 \
    libavformat58 \
    libavcodec58 \
    libswresample3 \
    libswscale5 \
    && rm -rf /var/lib/apt/lists/*

# 시스템 종속성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 종속성 설치
COPY ./requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# 애플리케이션 코드 복사
COPY app/ .

# 애플리케이션 실행 명령
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
