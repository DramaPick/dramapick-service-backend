# 1️⃣ Python 3.10 기반 이미지 사용
FROM python:3.10

# 2️⃣ 작업 디렉토리 설정
WORKDIR /app

# 3️⃣ `requirements.txt` 복사 후 패키지 설치
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# 4️⃣ 전체 프로젝트 코드 복사
COPY . /app/

# 5️⃣ FastAPI 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
