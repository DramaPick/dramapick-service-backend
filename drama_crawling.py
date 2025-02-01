import redis
import json
import requests
from lxml import html

# REDIS 연결 설정
redis_client = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)

def search_drama(drama_title: str):
    # Redis에 캐시 데이터가 있는지 확인
    if redis_client.exists(drama_title):
        cached_data = redis_client.get(drama_title)
        return json.loads(cached_data)

    # 네이버 검색 URL
    search_url = f"https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query={drama_title}"

    # User-Agent 설정
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
    }

    # 크롤링 요청
    response = requests.get(search_url, headers=headers)
    tree = html.fromstring(response.content)

    # 크롤링 데이터 추출 (XPath 사용)
    try:
        title = tree.xpath("//div[contains(@class, 'title_area')]/h2/span/strong/a/text()")[0].strip()  # 드라마 제목
        broadcaster = tree.xpath("//div[contains(@class, 'detail_info')]/dl/div[1]/dd/a/text()")[0].strip()  # 방송사
        air_date = tree.xpath("//div[contains(@class, 'detail_info')]/dl/div[1]/dd/span/text()")[0].strip()  # 방영일
    except IndexError:
        return None

    # 결과 반환
    drama_data = {
        "title": title,
        "broadcaster": broadcaster,
        "air_date": air_date,
    }
    redis_client.set(drama_title, json.dumps(drama_data))

    return drama_data

def get_drama(drama_title: str):
    # Redis에서 데이터 조회
    if redis_client.exists(drama_title):
        cached_data = redis_client.get(drama_title)
        return json.loads(cached_data)
    else:
        return None

def get_drama_person(drama_title: str):
    # 네이버 검색 URL
    search_url = f"https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query={drama_title}"

    # User-Agent 설정
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
    }

    # 크롤링 요청
    response = requests.get(search_url, headers=headers)
    tree = html.fromstring(response.content)

    # 크롤링 데이터 추출 (XPath 사용)
    try:
        person = tree.xpath("//div[contains(@class, 'scroll_box')]/div/div/ul/li[1]/div/a/text()")[0].strip()  # 드라마 제목
        broadcaster = tree.xpath("//div[contains(@class, 'detail_info')]/dl/div[1]/dd/a/text()")[0].strip()  # 방송사
        air_date = tree.xpath("//div[contains(@class, 'detail_info')]/dl/div[1]/dd/span/text()")[0].strip()  # 방영일
    except IndexError:
        return None