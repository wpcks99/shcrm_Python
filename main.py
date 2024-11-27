import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from predict import get_november_prediction, get_actual_november_data, load_or_train_model
from service import ImageProcessor
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from amount_of_use import get_monthly_data
import logging
import os
import asyncio

app = FastAPI()

logging.basicConfig(level=logging.INFO)

chrome_options = Options()
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Global driver initialization
driver = None

@app.on_event("startup")
async def startup():
    global driver
    try:
        # ChromeDriver 초기화
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        logging.info("ChromeDriver가 시작되었습니다.")
    except Exception as e:
        logging.error(f"ChromeDriver 시작 중 오류 발생: {str(e)}")

@app.on_event("shutdown")
async def shutdown():
    global driver
    if driver:
        driver.quit()
        logging.info("ChromeDriver가 종료되었습니다.")

CATEGORY_MAPPING = {
    '교통비': [
        '육상 여객 운송업', '항공 여객 운송업', '택시 운송업', 
        '주유소 운영업', '자동차 임대업', '시외버스 운송업'
    ],
    '소모품비': [
        '도매 및 소매업', '문구 소매업', '컴퓨터 및 주변기기 소매업', 
        '전기 용품 소매업'
    ],
    '복리후생비': [
        '음식점업', '일반 병원', '치과 병원', '레저업', '운동시설 운영업', 
        '화초 및 식물 소매업', '한식 음식점업', '중식 음식점업', 
        '일식 음식점업', '서양식 음식점업', '기타 외국식 음식점업', 
        '주점업', '다방업', '비알콜 음료점업', '호텔업', '기타 음식점업',
        '의약품 및 의료용품 소매업', '한식 일반 음식점업'
    ],
    '교육훈련비': [
        '기타 교육기관', '서적 소매업', '온라인 교육 서비스업', 
        '서적 및 문구용품 소매업', '서적', '신문 및 잡지류 소매업','서적, 신문 및 잡지류 소매업'
    ],
    '관리비': [
        '부동산 임대업', '자동차 임대업', '산업용 기계 및 장비 임대업', 
        '컴퓨터 및 주변기기 소매업', '신용카드업', '여신 금융업', 
        '기타 금융지원 서비스업', '기타 개인 서비스업', '유선 통신업', 
        '무선 통신업', '우편업', '택배업'
    ]
}

def get_category_from_detailed_category(detailed_category):
    category_mapping = {
        '교통비': 2,
        '소모품비': 5,
        '복리후생비': 3,
        '교육훈련비': 4,
        '관리비': 1
    }
    
    for category, keywords in CATEGORY_MAPPING.items():
        if detailed_category in keywords:
            category_name = category
            category_id = category_mapping.get(category_name, None)
            return {"categoryId": category_id, "categoryName": category_name}
    
    return {"categoryId": None, "categoryName": "기타"}

@app.post("/extract")
async def extract_data(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())  # 파일 저장
            image_path = temp_file.name

        # Selenium driver 인스턴스는 이미 startup에서 초기화됨
        processor = ImageProcessor(driver)

        # 비동기적으로 OCR 처리 및 카테고리 키워드 추출
        info_dict, img_url = await processor.process_image(image_path)
        business_numbers = info_dict.get("사업자번호", [])

        # 카테고리 키워드 추출 (비동기)
        category_keywords = await processor.extract_category_keywords(business_numbers)

        # 최종 결과 생성
        result = {
            "OCR_결과": {
                "사업자번호": info_dict.get("사업자번호", []),
                "가맹점명": info_dict.get("가맹점명", []),
                "거래일시": info_dict.get("거래일시", []),
                "금액": info_dict.get("금액", 0)
            },
            "카테고리_키워드": category_keywords or {},  # 카테고리 키워드가 없을 경우 빈 딕셔너리
            "이미지_URL": img_url
        }

        # 세세분류를 기반으로 최상위 카테고리 설정
        for biz_num, keywords in category_keywords.items():
            detailed_category = keywords.get("세세분류")
            if detailed_category:
                category = get_category_from_detailed_category(detailed_category)
                result["카테고리"] = category  # 카테고리 정보만 담도록 설정

        os.remove(image_path)  # 처리 후 파일 삭제
        logging.info(f"파일 '{image_path}' 처리 완료 및 삭제됨.")

        return JSONResponse(content=result)

    except Exception as e:
        logging.error(f"오류 발생: {e}")
        raise HTTPException(status_code=500, detail="데이터 추출에 실패했습니다.")

@app.post("/predict")
async def predict_consumption():
    logging.info("예측 요청 수신됨")
    
    try:
        model, df_scaled, scaler, df_pivot = load_or_train_model()
        
        if model is None:
            logging.error("모델이 생성되지 않음. 데이터가 부족할 수 있음.")
            return JSONResponse(content={"error": "모델을 생성할 데이터가 부족합니다."}, status_code=500)
        
        logging.info("모델 로드 또는 학습 완료")
        
        predicted_data = await asyncio.to_thread(get_november_prediction, model, df_scaled, scaler, df_pivot)
        actual_data = await asyncio.to_thread(get_actual_november_data)
        
        result = {
            "예측_사용량": predicted_data,
            "실제_11월_사용량": actual_data
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logging.error(f"예측 처리 중 오류 발생: {str(e)}")
        return JSONResponse(content={"error": f"예측 처리 중 오류가 발생했습니다. 상세: {str(e)}"}, status_code=500)

@app.post("/monthly_totals")
async def monthly_totals():
    try:
        result_data = await asyncio.to_thread(get_monthly_data)
        
        if not result_data:
            logging.warning("월별 사용 데이터가 없습니다.")
            raise HTTPException(status_code=404, detail="월별 사용 데이터가 없습니다.")
        
        logging.info(f"월별 사용 데이터: {result_data}")
        return JSONResponse(content=result_data)
    
    except HTTPException as http_error:
        logging.error(f"HTTP 예외 발생: {http_error.detail}")
        raise http_error
    except Exception as e:
        logging.error(f"월별 사용량 처리 중 오류 발생: {str(e)}")
        return JSONResponse(content={"error": f"월별 사용량 처리 중 오류가 발생했습니다. 상세: {str(e)}"}, status_code=500)
