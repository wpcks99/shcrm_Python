import asyncio
import os
import uuid
from PIL import Image, ImageDraw
import cv2
import easyocr
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager  # webdriver-manager 임포트 추가

class ImageProcessor:
    def __init__(self, driver):
        # easyocr Reader와 ChromeDriver 초기화
        self.reader = easyocr.Reader(['ko', 'en'], gpu=True)
        self.driver = driver  # 외부에서 전달받은 driver 사용

    async def process_image(self, img_path):
        """이미지 처리 및 OCR, 정보 추출"""
        # 이미지 파일 읽기 및 텍스트 추출 비동기 처리
        image = await asyncio.to_thread(cv2.imread, img_path)
        if image is None:
            raise Exception(f"이미지를 읽을 수 없습니다: {img_path}")

        # 텍스트 추출
        results = await asyncio.to_thread(self.reader.readtext, image)
        extracted_text = []
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        for detection in results:
            bbox = detection[0]
            text = detection[1]
            extracted_text.append(text)

        # OCR 결과를 사용하여 정보 추출
        info_dict = {
            "사업자번호": self.extract_business_numbers(extracted_text),
            "가맹점명": self.extract_store_names(extracted_text),
            "거래일시": self.extract_transaction_date(extracted_text),
            "금액": self.extract_max_number(extracted_text)
        }

        # 이미지 파일을 로컬 저장 후 경로 반환
        img_url = await self.save_image_locally(img, img_path)

        return info_dict, img_url

    def extract_business_numbers(self, text_list):
        """사업자 번호 추출"""
        business_number_pattern = re.compile(r'\b\d{3}-?\d{2}-?\d{5}\b')
        return [match for text in text_list for match in business_number_pattern.findall(text)]

    def extract_store_names(self, text_list):
        """가맹점명 추출"""
        store_name_pattern = re.compile(r'(매장명|상호명|회사명|업체명|가맣점명|[상싱성][호오]|[회훼]사)\s*[:;：]?\s*([^\s)]+(?:\s*\S*)*?[점]\s*?\S*)')
        return [match[1] for text in text_list for match in store_name_pattern.findall(text)]

    def extract_transaction_date(self, text_list):
        """거래 일시 추출"""
        date_pattern = re.compile(r'([거기][래레][일닐]|[결겔]제[일닐]|거[래레]일시|[결겔]제날짜|날짜|일자)?\s*[:;：]?\s*' r'(\d{4}[-/.]\d{2}[-/.]\d{2}|\d{2}[-/.]\d{2}[-/.]\d{4}|\d{2}[-/.]\d{2}[-/.]\d{2})')
        extracted_dates = []

        for text in text_list:
            matches = date_pattern.findall(text)

            for match in matches:
                date_part = match[1]
                if self.is_valid_date(date_part):
                    extracted_dates.append(date_part)

        return extracted_dates

    def extract_max_number(self, text_list):
        """가장 큰 숫자 추출"""
        all_numbers = []
        for text in text_list:
            numbers = re.findall(r'\d{1,3}(?:,\d{3})*', text)
            numbers = [int(num.replace(',', '')) for num in numbers]
            all_numbers.extend(numbers)
        return max(all_numbers, default=0) if all_numbers else None

    def is_valid_date(self, date_str):
        """날짜 유효성 검사"""
        parts = re.split(r'[-/.]', date_str)
        if len(parts) == 3:
            if len(parts[0]) == 4 and len(parts[1]) == 2 and len(parts[2]) == 2:
                year, month, day = map(int, parts)
                return 2000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31
            elif len(parts[0]) == 2 and len(parts[1]) == 2 and len(parts[2]) == 4:
                day, month, year = map(int, parts)
                return 2000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31
            elif len(parts[0]) == 2 and len(parts[1]) == 2 and len(parts[2]) == 2:
                month, day, year = map(int, parts)
                year += 2000 if year < 100 else 0
                return 2000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31
        return False

    async def extract_category_keywords(self, business_numbers):
        """카테고리 키워드 추출"""
        category_keywords_dict = {}

        for business_number in business_numbers:
            business_number_clean = business_number.replace("-", "")
            
            address = 'https://bizno.net/article/' + business_number_clean
            print(f"접속 중인 URL: {address}")

            self.driver.get(address)

            # 상호명 추출
            try:
                shop_name = self.driver.find_element(By.XPATH, '/html/body/section[2]/div/div/div[1]/div[1]/div/div[1]/div/a/h1').text
            except NoSuchElementException:
                shop_name = "상호명 없음"  
            
            # category_keywords 추출
            category_keywords = None
            try:
                element = self.driver.find_element(By.XPATH, '/html/body/section[2]/div/div/div[1]/div[1]/div/table/tbody/tr[2]/td')
                if all(label in element.text for label in ["대분류", "중분류", "소분류", "세분류", "세세분류"]):
                    category_keywords = element.text
            except NoSuchElementException:
                pass
            
            if not category_keywords:
                try:
                    element = self.driver.find_element(By.XPATH, '/html/body/section[2]/div/div/div[1]/div[1]/div/table/tbody/tr[4]/td')
                    if all(label in element.text for label in ["대분류", "중분류", "소분류", "세분류", "세세분류"]):
                        category_keywords = element.text
                except NoSuchElementException:
                    pass
            
            if category_keywords:
                print(f"업태: {category_keywords}")
            
                category_dict = {
                    "상호명": shop_name,
                    "대분류": re.search(r"대분류\s*:\s*(.*?)(?=\s*중분류|$)", category_keywords),
                    "중분류": re.search(r"중분류\s*:\s*(.*?)(?=\s*소분류|$)", category_keywords),
                    "소분류": re.search(r"소분류\s*:\s*(.*?)(?=\s*세분류|$)", category_keywords),
                    "세분류": re.search(r"세분류\s*:\s*(.*?)(?=\s*세세분류|$)", category_keywords),
                    "세세분류": re.search(r"세세분류\s*:\s*(.*)", category_keywords)
                }

                for key in category_dict:
                    if isinstance(category_dict[key], re.Match):
                        category_dict[key] = category_dict[key].group(1).strip()
                    
                    elif category_dict[key] is not None:
                        category_dict[key] = category_dict[key].strip()
                
                category_keywords_dict[business_number] = category_dict
            else:
                print(f"사업자 번호 {business_number}에 대한 정보를 찾을 수 없습니다.")

        return category_keywords_dict

    async def save_image_locally(self, image, original_image_path):
        """이미지 로컬 저장"""
        # 원본 파일의 확장자 추출
        _, file_extension = os.path.splitext(original_image_path)
        
        # 경로를 C:\Users\Yeon Je Chan\Desktop\shcrm로 설정
        save_path = os.path.join(r'C:\Users\Yeon Je Chan\Desktop\shcrm', str(uuid.uuid4()) + file_extension)

        # 디렉토리가 존재하지 않으면 생성
        if not os.path.exists(r'C:\Users\Yeon Je Chan\Desktop\shcrm'):
            os.makedirs(r'C:\Users\Yeon Je Chan\Desktop\shcrm')

        # 이미지 저장
        image.save(save_path)

        # URL 생성
        file_url = f"http://shcrm.ddns.net:8080/file/{os.path.basename(save_path)}"
        return file_url
