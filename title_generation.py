from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import os

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
client = OpenAI(api_key=os.getenv("GPT_API_KEY"))

def generate_highlight_title(input_title: str) -> List[str]:
    """
    입력된 내용을 기반으로 제목 5가지를 추천합니다.

    :param input_title: 사용자가 원하는 내용
    :return: 추천된 제목 리스트
    """
    prompt = f"""
    다음은 일반적인 유튜브 쇼츠 하이라이트 제목의 예시입니다.

    예시:
    - 어설픈 중국 부자 연기에 웃참하는 재벌 가족ㅋㅋㅋ
    - 농촌 체험한 재벌 불면증 싹 사라짐ㅋㅋㅋ
    - 어딜 내놔도 인기 폭발 남편 단속하는 와이프

    위와 같은 스타일로, 띄어쓰기를 포함하여 25글자 이내의 유튜브 쇼츠 제목 5개를 추천해주세요.

    💡 유의사항:
    - 제목은 짧고 임팩트 있게 작성해주세요.
    - 기준 제목의 **핵심 동작(동사)과 동작의 대상(명사)을 유지해야 합니다.**
    - **동사의 방식(형용사로, 멋지게, 극적으로, 로맨틱하게 등 혹은 비유적 표현)은 자유롭게 표현해도 됩니다.**
    - 기준 제목의 의미를 벗어나지 마세요.
    - **이모티콘은 절대 사용하지 마세요.**

    📌 기준 제목: "{input_title}"

    위의 유의사항을 명시해 새로운 쇼츠 제목 5가지를 추천해주세요.
    """

    # OpenAI API를 사용하여 제목 생성
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 사람들이 주목할 만한 컨텐츠를 기획하는 감독입니다. 드라마의 하이라이트 장면을 유튜브 숏폼으로 제작할 때, 상단에 삽입될 쇼츠 제목을 작성해야 합니다."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )

    # 생성된 제목 리스트 반환
    titles = [choice.message.content.strip() for choice in response.choices]
    for i in range(len(titles)):
        title_ = f'\n{input_title}'
        titles[i] += title_
    return titles