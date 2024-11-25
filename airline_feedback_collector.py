import os
import time
import json
import requests
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import logging
from datetime import datetime

# 프로젝트 이름
PROJECT_NAME = "AirlineFeedback"

# 로그 파일 이름 설정
LOG_FILE = f"{PROJECT_NAME}_{datetime.now().strftime('%Y-%m-%d')}.log"

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # 전체 로그 레벨 설정
    format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",  # 메시지 포맷 설정
    handlers=[
        logging.FileHandler(LOG_FILE),  # 파일 핸들러
        logging.StreamHandler()  # 콘솔 핸들러
    ]
)

# 로거 생성
logger = logging.getLogger(PROJECT_NAME)

# 환경 변수 로드
load_dotenv()

# 환경 변수에서 API 키 읽기
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")


class FeedbackCollector(ABC):
    """
    Abstract Base Class for Feedback Collectors
    """

    @abstractmethod
    def fetch_data(self, query, **kwargs):
        """
        Abstract method to fetch data.
        """
        pass

    @staticmethod
    def save_to_json(data, filename):
        """
        Save collected data to a JSON file.
        """
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


class TwitterCollector(FeedbackCollector):
    """
    Collector for Twitter Data
    """

    def __init__(self, bearer_token):
        self.bearer_token = bearer_token
        self.base_url = "https://api.twitter.com/2/tweets/search/recent"

    def fetch_data(self, query, max_results=10, next_token=None):        
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        params = {
            "query": query,
            "tweet.fields": "id,text,author_id,created_at",
            "max_results": max_results,
        }
        if next_token:  # next_token이 있으면 파라미터에 추가
            params["next_token"] = next_token        
        response = requests.get(self.base_url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Twitter API Error {response.status_code}: {response.json()}")            
            return None


class NaverCollector(FeedbackCollector):
    """
    Base Collector for Naver Blog and Cafe Data
    """

    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret

    def fetch_data(self, query, display=10, start=1, endpoint=""):
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }
        params = {
            "query": query,
            "display": display,
            "start": start,
        }
        response = requests.get(endpoint, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Naver API Error {response.status_code}: {response.json()}")            
            return None


class NaverBlogCollector(NaverCollector):
    """
    Collector for Naver Blog Data
    """

    def __init__(self, client_id, client_secret):
        super().__init__(client_id, client_secret)
        # 네이버 검색의 블로그 검색 결과를 반환합니다.
        self.endpoint = "https://openapi.naver.com/v1/search/blog.json"

    def fetch_data(self, query, display=10, start=1):
        return super().fetch_data(query, display, start, endpoint=self.endpoint)


class NaverCafeCollector(NaverCollector):
    """
    Collector for Naver Cafe Data
    """

    def __init__(self, client_id, client_secret):
        super().__init__(client_id, client_secret)
        # 네이버 검색의 카페글 검색 결과를 반환합니다.
        self.endpoint = "https://openapi.naver.com/v1/search/cafearticle.json"

    def fetch_data(self, query, display=10, start=1):
        return super().fetch_data(query, display, start, endpoint=self.endpoint)


def fetch_and_process_data(collector, query, max_requests, delay, **kwargs):
    """
    데이터 수집 및 처리 공통 로직
    :param collector: 데이터 수집기 객체
    :param query: 검색 쿼리
    :param max_requests: 최대 요청 횟수
    :param delay: 요청 간 대기 시간
    :param kwargs: 추가 파라미터 (pagination 관련 등)
    """
    logger.info(f"Fetching data with {collector.__class__.__name__}...")
    start_index = kwargs.get("start_index", 1)
    next_token = kwargs.get("next_token", None)
    display = kwargs.get("display", 100)
    data_type = kwargs.get("data_type", "data")

    for request_count in range(max_requests):
        try:
            # 데이터 수집
            # Fetch data with pagination for Twitter or Naver (Blog or Cafe)
            # Adjusts logic based on the collector type (Twitter uses next_token; Naver uses start_index)
            logger.info(f"[{collector.__class__.__name__}] Fetching data ...")                  
            if isinstance(collector, TwitterCollector):
                # For Twitter: Use next_token for pagination
                data = collector.fetch_data(query, max_results=100, next_token=next_token)
            else:
                # For Naver (Blog or Cafe): Use start_index for pagination
                data = collector.fetch_data(query, display=display, start=start_index)

            # 데이터 검증
            if data:
                items = data.get(data_type, [])
                retrieved_items = len(items)
                logger.info(f"[{collector.__class__.__name__}] Request {request_count + 1}: Retrieved {retrieved_items} items")

                # 데이터가 없을 경우 중단
                if retrieved_items == 0:
                    logger.warning(f"[{collector.__class__.__name__}] No {data_type} retrieved. Stopping further requests.")
                    break

                # 데이터 저장
                class_name = collector.__class__.__name__.lower()
                if "twitter" in class_name:
                    file_prefix = "twitter"
                elif "blog" in class_name:
                    file_prefix = "naver_blog"
                elif "cafe" in class_name:
                    file_prefix = "naver_cafe"
                else:
                    file_prefix = class_name  # 기본적으로 클래스 이름 그대로 사용
                output_file = f"./data/{file_prefix}_data_page_{request_count + 1}.json"
                collector.save_to_json(data, output_file)
                logger.info(f"Saved data to {output_file}")

                # 페이지네이션 업데이트
                if "next_token" in kwargs:
                    next_token = data.get("meta", {}).get("next_token")
                    if not next_token:
                        logger.info(f"[{collector.__class__.__name__}] No more data available.")
                        break
                elif "start_index" in kwargs:
                    start_index += display
                    if start_index > 1000:
                        logger.info(f"[{collector.__class__.__name__}] Naver API pagination limit reached (max 1000).")
                        break  # 네이버 API의 최대 시작 인덱스 한도 초과
                    # start 제한으로 인해 최대 1,000개의 데이터만 접근 가능
                    # 페이지네이션으로 접근 가능한 데이터의 범위를 제한
                    # 25,000회의 일일 호출 횟수 제한과는 무관
                    # 일일 호출 횟수가 25,000회를 초과하면 HTTP 429 (Too Many Requests) 응답이 반환되어 요청이 차단됨.                    
            else:
                logger.warning(f"[{collector.__class__.__name__}] Request {request_count + 1}: No response received.")
                break  # Stop if response is invalid
            # Delay between requests to avoid hitting rate limits
            time.sleep(delay)  # 요청 간 대기
        except Exception as e:
            if "429" in str(e):  # Handle rate limit specifically
                logger.error(f"[{collector.__class__.__name__}] Rate limit exceeded (429). Retrying after delay...")
                time.sleep(60)  # Wait longer before retrying
            else:
                logger.error(f"[{collector.__class__.__name__}] Error during request {request_count + 1}: {e}")
            break


if __name__ == "__main__":
    # Ensure the data directory exists
    os.makedirs("./data", exist_ok=True)

    # Initialize collectors
    twitter_collector = TwitterCollector(TWITTER_BEARER_TOKEN)
    naver_blog_collector = NaverBlogCollector(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)
    naver_cafe_collector = NaverCafeCollector(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)

    # Define search queries
    twitter_query = "(대한항공 후기) lang:ko"
    naver_blog_query = "대한항공 후기"
    naver_cafe_query = "대한항공 후기"
    # twitter_query = input("Enter Twitter search query: ")
    # naver_blog_query = input("Enter Naver Blog search query: ")
    # naver_cafe_query = input("Enter Naver Cafe search query: ")

    # Fetch Twitter data
    fetch_and_process_data(
        twitter_collector,
        twitter_query,
        max_requests=1,   # 최대 요청 수
        delay=40,         # 요청 간 대기 시간 (36[초/request] = 15[분] / 25[request] * 60[초/분])
        next_token=None,  # Pagination 토큰
        data_type="data",
    )

    # Fetch Naver Blog data
    fetch_and_process_data(
        naver_blog_collector,
        naver_blog_query,
        max_requests=1,   # 최대 요청 수
        delay=1,          # 요청 간 대기 시간 (1초로 설정, 네이버 API 호출 한도에 맞춤)
        start_index=1,    # Pagination 시작 인덱스
        display=100,      # 한 번에 가져올 데이터 개수 (최대 100)
        data_type="items",
        # 초당 요청 횟수가 10회를 초과하면 HTTP 429 (Too Many Requests) 오류가 발생할 수 있습니다.        
        # 초당 최대 10회 요청이 가능하므로 호출 간 최소 대기 시간은 0.1초입니다.
        # 안정적인 호출을 위해 초당 1회의 요청 간격(1초 대기)을 설정하는 것이 일반적입니다.        
    )

    # Fetch Naver Cafe data
    fetch_and_process_data(
        naver_cafe_collector,
        naver_cafe_query,
        max_requests=1,   # 최대 요청 수
        delay=1,          # 요청 간 대기 시간 (1초로 설정, 네이버 API 호출 한도에 맞춤)
        start_index=1,    # Pagination 시작 인덱스
        display=100,      # 한 번에 가져올 데이터 개수 (최대 100)
        data_type="items",
        # 초당 요청 횟수가 10회를 초과하면 HTTP 429 (Too Many Requests) 오류가 발생할 수 있습니다.        
        # 초당 최대 10회 요청이 가능하므로 호출 간 최소 대기 시간은 0.1초입니다.
        # 안정적인 호출을 위해 초당 1회의 요청 간격(1초 대기)을 설정하는 것이 일반적입니다.              
    )

    logger.info("Data collection completed!")
