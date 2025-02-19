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
    # OpenAI API를 사용하여 제목 생성
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 사람들이 주목할 만한 컨텐츠를 기획하는 감독입니다. 드라마의 하이라이트 장면을 유튜브 숏폼으로 제작할 때, 위에 들어갈 제목을 작성해야 합니다."},
            {"role": "user", "content": f"다음은 사용자가 원하는 내용입니다. 이를 기반으로, 유튜브 쇼츠 하이라이트의 제목 5가지를 추천해 주세요. 다음은 제목의 예시들입니다. '어설픈 중국 부자 연기에 웃참하는 재벌 가족ㅋㅋㅋ', '농촌 체험한 재벌 불면증 싹 사라짐ㅋㅋㅋ', '어딜 내놔도 인기 폭발 남편 단속하는 와이프':\n\n{input_title}\n\n제목 5가지 추천:"}
        ],
        max_tokens=300,
        temperature=0.7
    )

    # 생성된 제목 리스트 반환
    titles = [choice.message.content.strip() for choice in response.choices]
    return titles


# input_title = "기억 상실에 걸린 홍해인을 회유하는 백현우. 감동적이고 슬픈 분위기."
# recommended_titles = generate_highlight_title(input_title)

# for idx, title in enumerate(recommended_titles, start=1):
#    print(f"{title}")