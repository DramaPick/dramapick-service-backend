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
    yasm \
    nasm \
    git \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

# FFmpeg 소스 다운로드 및 빌드
RUN git clone --depth 1 https://git.ffmpeg.org/ffmpeg.git /ffmpeg \
    && cd /ffmpeg \
    && ./configure --prefix=/usr/local --enable-shared --disable-static \
    && make -j$(nproc) \
    && make install

# LD_LIBRARY_PATH 환경 변수에 라이브러리 경로 추가
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# FFmpeg 바이너리 확인
RUN ffmpeg -version

# Python 종속성 설치
COPY ./requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# 애플리케이션 코드 복사
COPY app/ .

# 애플리케이션 실행 명령
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]