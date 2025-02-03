from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import os
from s3_client import s3_client
from botocore.exceptions import NoCredentialsError
from drama_crawling import search_drama, get_drama
from fastapi import HTTPException

# S3 설정
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "test-fastapi-bucket")
S3_REGION_NAME = os.getenv("S3_REGION_NAME", "ap-northeast-2")


def search_drama_api(drama_title: str):
    result = search_drama(drama_title)
    print(f"🔍 검색 결과: {result}")  # 실제 검색 결과 확인
    if result:
        return {"status": "success", "data": result}
    else:
        raise HTTPException(status_code=404, detail="드라마 정보를 찾을 수 없습니다.")
    

def get_drama_api(drama_title: str):
    # Redis에서 데이터 조회
    result = get_drama(drama_title)
    if result:
        return result
    else:
        raise HTTPException(status_code=404, detail="드라마 정보를 찾을 수 없습니다.")
    
def upload_to_s3(file_path, s3_filename, bucket_name=S3_BUCKET_NAME):
    try:
        # 파일 업로드
        s3_client.upload_file(file_path, bucket_name, s3_filename)

        # 업로드된 파일의 URL 생성
        s3_url = f"https://{bucket_name}.s3.{S3_REGION_NAME}.amazonaws.com/{s3_filename}"

        print(f"✅ 업로드 완료: {s3_url}")
        return s3_url

    except FileNotFoundError:
        print("❌ 파일을 찾을 수 없습니다.")
        return None
    except NoCredentialsError:
        print("❌ AWS 인증 정보가 없습니다.")
        return None
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return None


def clip_text(local_path_list, task_id, drama_title):
    font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

    try:
        idx = 1
        url_list = []
        for local_path in local_path_list:
            video = VideoFileClip(local_path)

            drama_info = get_drama_api(drama_title)
            if not drama_info:
                drama_info = search_drama_api(drama_title)
                if not drama_info:
                    raise ValueError("❌ 드라마 정보를 찾을 수 없습니다.")

            text1 = drama_title.encode('utf-8')
            txt_clip = TextClip(text1, fontsize=55, color='white', font=font_path)
            txt_clip = txt_clip.set_position(('center', video.h - 450)).set_duration(video.duration)

            text2 = f"{drama_info['broadcaster']} - {drama_info['air_date']}"
            txt_clip2 = TextClip(text2, fontsize=30, color='white', font=font_path)
            txt_clip2 = txt_clip2.set_position(('center', video.h - 350)).set_duration(video.duration)

            TEMP_DIR = "tmp"
            filename = f"{task_id}_highlight_with_info_{idx}.mp4"
            output_path = os.path.join(TEMP_DIR, filename)  # 임시 파일 경로 설정
            
            result = CompositeVideoClip([video, txt_clip, txt_clip2])
            result.write_videofile(output_path)

            s3_url_ = upload_to_s3(output_path, filename)
            url_list.append(s3_url_)

            print(f"Successfully saved highlight {idx + 1} in S3 BUCKET!!")

            os.remove(local_path)
            os.remove(output_path)

            idx += 1
        return url_list
    
    except ValueError as e:
        return f"🚨 오류: {e}"
    except Exception as e:
        return f"🚨 예기치 않은 오류 발생: {e}"
