from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ImageClip
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


def crop_and_pad_to_1080x1920(clip):
    clip = clip.resize(height=960)

    new_width, _ = clip.size
    if new_width < 1080:
        clip = clip.resize(width=1080)

    _, final_height = clip.size
    if final_height < 1920:
        clip = clip.on_color(size=(1080, 1920), color=(0, 0, 0), pos=('center', 'center'))
    
    return clip

def clip_and_save_highlights(local_path, task_id, drama_title, adjusted_highlights):
    TEMP_DIR = 'tmp'
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

    try:
        video = VideoFileClip(local_path)
        
        drama_info = get_drama_api(drama_title)
        if not drama_info:
            drama_info = search_drama_api(drama_title)
            if not drama_info:
                raise ValueError("❌ 드라마 정보를 찾을 수 없습니다.")

        text1 = drama_title.encode('utf-8')
        txt_clip1 = TextClip(text1, fontsize=55, color='white', font=font_path)
        txt_clip1.save_frame("tmp/img_clip1.png", t=0)

        text2 = f"{drama_info['broadcaster']} - {drama_info['air_date']}"
        txt_clip2 = TextClip(text2, fontsize=30, color='white', font=font_path)
        txt_clip2.save_frame("tmp/img_clip2.png", t=0)

        url_list = []
        for idx, (start, end) in enumerate(adjusted_highlights):
            highlight_clip = video.subclip(start, end)

            highlight_clip = crop_and_pad_to_1080x1920(highlight_clip)

            filename = f"{task_id}_highlight_with_info_{idx+1}.mp4"
            output_path = os.path.join(TEMP_DIR, filename)  # 임시 파일 경로 설정
            
            text_img1 = ImageClip("img_clip1.png").set_duration(highlight_clip.duration).set_position(('center', highlight_clip.h - 450))
            text_img2 = ImageClip("img_clip2.png").set_duration(highlight_clip.duration).set_position(('center', highlight_clip.h - 350))
        
            result = CompositeVideoClip([highlight_clip, text_img1, text_img2])
            # result.write_videofile(output_path, codec="libx264", audio_codec="aac", preset="ultrafast")
            result.write_videofile(output_path, codec="libx264", audio_codec="aac", threads=8, fps=24)

            s3_url_ = upload_to_s3(output_path, filename)
            url_list.append(s3_url_)

            print(f"Successfully saved highlight {idx + 1} in S3 BUCKET!!")

            os.remove(local_path)
            os.remove(output_path)

        os.remove("tmp/img_clip1.png")
        os.remove("tmp/img_clip2.png")

        return url_list
    
    except ValueError as e:
        return f"🚨 오류: {e}"
    except Exception as e:
        return f"🚨 예기치 않은 오류 발생: {e}"

if __name__ == "__main__":
    clip_and_save_highlights("시연용비디오.mov", 1111, "눈물의 여왕", [[10.0, 50.0]])