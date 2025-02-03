import pysrt
from datetime import datetime
import subprocess
import whisper
import cv2
from moviepy.editor import VideoFileClip
import os
import warnings
from datetime import datetime
from google.cloud import speech
import io


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "dramapickstt-4efdec08fb9c.json"
warnings.filterwarnings('ignore')


def parse_time(srt_time_str):
    """SRT 시간 문자열을 timedelta로 변환"""
    return datetime.strptime(srt_time_str, "%H:%M:%S,%f")


def time_to_srt_format(time):
    """timedelta를 SRT 시간 문자열로 변환"""
    return str(time).split('.')[0].replace(',', '.')


def transcribe_audio(audio_path):
    """Google STT API를 사용하여 오디오 파일을 텍스트로 변환"""
    client = speech.SpeechClient()

    with io.open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ko-KR",
    )

    response = client.recognize(config=config, audio=audio)

    transcripts = []
    for result in response.results:
        transcripts.append(result.alternatives[0].transcript)

    return transcripts


def merge_srt_lines(input_srt_path, output_srt_path, min_gap=1.0):
    """
    SRT 파일에서 두 자막 간의 시간 차이를 계산하고, 
    시간 차이가 min_gap 이하인 자막들을 합침.

    :param input_srt_path: 입력 SRT 파일 경로
    :param output_srt_path: 출력 SRT 파일 경로
    :param min_gap: 자막 간 간격이 min_gap 이하일 경우 합침
    """
    # Google STT API를 사용하여 오디오 파일을 텍스트로 변환
    transcripts = pysrt.open(input_srt_path)
    # transcripts = transcribe_audio(input_srt_path)

    # 자막 합치기
    merged_subs = []
    current_sub = None
    for i in range(len(transcripts)):
        sub = transcripts[i]
        if current_sub is None:
            current_sub = sub
            continue

        # 자막 간의 간격 계산
        time_gap = (parse_time(sub.start.to_time().strftime("%H:%M:%S,%f")) -
                    parse_time(current_sub.end.to_time().strftime("%H:%M:%S,%f"))).total_seconds()

        if time_gap < min_gap:
            # 간격이 min_gap 이하이면 자막을 합침
            current_sub.text += ' ' + sub.text
            current_sub.end = sub.end
        else:
            # 자막 간격이 min_gap 초 이상이면 새로운 자막으로 저장
            merged_subs.append(current_sub)
            current_sub = sub

    # 마지막 자막을 추가
    if current_sub is not None:
        merged_subs.append(current_sub)

    # 합쳐진 자막 저장
    with open(output_srt_path, 'w', encoding='utf-8') as f:
        for index, sub in enumerate(merged_subs, start=1):
            f.write(f"{index}\n")
            f.write(f"{sub.start} --> {sub.end}\n")
            f.write(f"{sub.text}\n\n")

    print("---------- Merged ----------")


def extract_audio(video_path, audio_output_path):
    """
    FFmpeg를 사용해 비디오에서 오디오를 추출.
    :param video_path: 비디오 파일 경로 (예: 'video.mov')
    :param audio_output_path: 저장할 오디오 파일 경로 (예: 'audio.wav')
    """
    # 정확한 FFmpeg 명령어로 수정
    command = [
        "ffmpeg",         # ffmpeg 실행 파일
        "-i", video_path,  # 입력 파일
        "-vn",            # 비디오 트랙 제외
        "-acodec", "pcm_s16le",  # 오디오 코덱 (16-bit PCM)
        "-ar", "16000",   # 샘플링 레이트 16kHz
        "-ac", "1",       # 모노 오디오 (1 채널)
        audio_output_path  # 출력 파일 경로 (예: 'output_audio.wav')
    ]

    try:
        # subprocess를 사용하여 명령어 실행
        subprocess.run(command, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Audio extraction successful! Saved to {audio_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e.stderr.decode()}")


def generate_srt_from_audio(audio_path, srt_output_path):
    model = whisper.load_model("base")  # Whisper 모델 로드

    # 오디오 파일을 텍스트로 변환 (타임스탬프 포함)
    result = model.transcribe(audio_path, word_timestamps=True)

    with open(srt_output_path, 'w', encoding='utf-8') as f:
        index = 1
        for segment in result['segments']:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']

            # 자막 번호, 시간 범위, 텍스트 형식으로 SRT 파일 작성
            start_time_str = format_time(start_time)
            end_time_str = format_time(end_time)

            f.write(f"{index}\n")
            f.write(f"{start_time_str} --> {end_time_str}\n")
            f.write(f"{text}\n\n")
            index += 1

    print(f"COMPLETE CREATING SRT FILE")


def format_time(seconds):
    """초를 'HH:MM:SS,SSS' 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_int = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds_int:02},{milliseconds:03}"


def convert_to_seconds(time_str):
    """
    'HH:MM:SS,SSS' 형식의 시간을 초로 변환.
    예: "00:00:10,500" -> 10.5초
    """
    time_obj = datetime.strptime(time_str, "%H:%M:%S,%f")
    return time_obj.second + time_obj.minute * 60 + time_obj.hour * 3600 + time_obj.microsecond / 1e6


def find_end_time_after(audio_txt_path, start_time_sec, flag):
    """
    주어진 초 이후에 등장인물이 대사를 마친 시점을 찾는 함수.

    :param audio_txt_path: 대사 내용이 저장된 .srt 파일 경로
    :param start_time_sec: 대사를 시작할 초 (seconds)
    :return: 해당 시점 이후 대사가 끝난 시점 초
    """
    end_time = None  # 대사 끝나는 시간 (초 단위)

    with open(audio_txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # .srt 파일 파싱: 짝수 번째 줄은 시간 정보, 홀수 번째 줄은 대사 내용
    for i in range(len(lines)):
        time_range = lines[i].strip()  # '00:00:01,000 --> 00:00:05,000'
        if '-->' in time_range:
            start_time_str, end_time_str = time_range.split('-->')
            print(f"-- end_time_str : {end_time_str} --")
        else:
            continue
        end_time_str = end_time_str.strip()

        # 시작과 끝 시간 변환
        end_time_sec = convert_to_seconds(end_time_str)

        if flag == "s":
            if start_time_sec < end_time_sec and end_time_sec <= start_time_sec+20:
                if end_time is None or end_time_sec < end_time:
                    end_time = end_time_sec
                    break
        elif flag == "e":
            # 주어진 start_time_sec 이후에 끝나는 첫 번째 대사 종료 시간 찾기
            if start_time_sec < end_time_sec:
                if end_time is None or end_time_sec < end_time:
                    print(f"===== end_time_sec: {end_time_sec} =====")
                    end_time = end_time_sec
                    break

    if end_time is None:
        return start_time_sec

    return end_time


def scene_detection(local_path, highlights):
    cap = cv2.VideoCapture(local_path)

    if not cap.isOpened():
        print(f"Error opening video source: {local_path}")
        exit(1)

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"TOTAL FRAMES - {total_frames}")
    except ValueError:
        total_frames = -1

    res, frame = cap.read()
    if not res:
        print("Error reading the first frame from the video.")
        exit(1)

    previous_histogram = cv2.calcHist([frame], [0, 1, 2], None, [
                                      8, 8, 8], [0, 256, 0, 256, 0, 256])
    previous_histogram = cv2.normalize(
        previous_histogram, previous_histogram).flatten()

    audio_path = f"audio.wav"
    extract_audio(local_path, audio_path)
    dialoges_srt_path = "drama_dialogues.srt"
    generate_srt_from_audio(audio_path, dialoges_srt_path)
    dialoges_final_path = "merged.srt"
    merge_srt_lines(dialoges_srt_path, dialoges_final_path, min_gap=1.0)

    adjusted_highlights = []
    for start, end in highlights:
        adjusted_start = find_end_time_after(
            dialoges_final_path, start - 15, flag="s")
        adjusted_end = find_end_time_after(dialoges_final_path, end, flag="e")
        # print(f"adjusted_end: {adjusted_end}")

        adjusted_highlights.append([adjusted_start, adjusted_end])

    print("Adjusted highlights:", adjusted_highlights)

    # Cleanup
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    os.remove(audio_path)
    os.remove(dialoges_srt_path)
    os.remove(dialoges_final_path)

    return adjusted_highlights


def crop_and_pad_to_1080x1920(clip):
    clip = clip.resize(height=960)
    # clip = resize.resize(clip, height=960)

    new_width, new_height = clip.size

    if new_width < 1080:
        clip = clip.resize(width=1080)
        # clip = resize.resize(clip, width=1080)

    final_width, final_height = clip.size

    if final_height < 1920:
        clip = clip.on_color(size=(1080, 1920), color=(0, 0, 0), pos=('center', 'center'))

    return clip

def save_highlights_with_moviepy(local_path, adjusted_highlights, task_id):
    TEMP_DIR = 'tmp'
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    video = VideoFileClip(local_path)


    local_path_list = []
    for idx, (start, end) in enumerate(adjusted_highlights):
        highlight_clip = video.subclip(start, end)

        highlight_clip = crop_and_pad_to_1080x1920(highlight_clip)

        filename = f"{task_id}_highlight_{idx + 1}.mp4"
        output_path = os.path.join(TEMP_DIR, filename)

        highlight_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print(f"Saved highlight {idx + 1} to {output_path}")

        local_path_list.append(output_path)

    video.close()
    
    return local_path_list
