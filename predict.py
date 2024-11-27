from fastapi import FastAPI
from fastapi.responses import JSONResponse
from datetime import datetime
import logging
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import json


# MySQL 연결 정보
host = "shcrm.ddns.net"
port = 5001
database = "shcrm"
user = "root"
password = "9868"

# SQLAlchemy 연결 URL
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")

# 모델 파일 경로
model_file = "lstm_model.h5"

# 모델 로드 또는 학습 함수
def load_or_train_model():
    today = datetime.now()
    today_str = today.strftime('%Y-%m-%d')  # 오늘 날짜를 'YYYY-MM-DD' 형식으로 변환

    # 예측에 필요한 데이터 준비
    query = f"""
        SELECT DATE_FORMAT(date, '%%Y-%%m') AS Month, category, SUM(amount) AS Amount
        FROM 사용금액
        WHERE date >= '2023-11-01' AND date <= '{today_str}'
        GROUP BY Month, category
        ORDER BY Month
    """
    df = pd.read_sql(query, engine)

    # 피벗 테이블 생성 (월별 계정과목별 총 금액)
    df_pivot = df.pivot_table(index='Month', columns='category', values='Amount', aggfunc='sum').fillna(0)

    # 스케일링 설정
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df_pivot)

    # 모델이 이미 존재하면 로드
    if os.path.exists(model_file):
        model = load_model(model_file)
        logging.info("기존 학습된 모델을 로드했습니다.")
        return model, df_scaled, scaler, df_pivot
    else:
        logging.info("모델이 존재하지 않아 새로운 모델을 학습합니다.")
        
        # 새로 모델을 학습
        def create_dataset(data, look_back=12):
            X, y = [], []
            for i in range(len(data) - look_back):
                X.append(data[i:(i + look_back), :])
                y.append(data[i + look_back, :])
            return np.array(X), np.array(y)

        look_back = 12
        X, y = create_dataset(df_scaled, look_back)

        # 훈련 데이터와 테스트 데이터 분리
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # 모델 생성
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=X_train.shape[1:]),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(y_train.shape[1])  # 각 계정과목별로 예측
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # 모델 훈련
        model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=0)

        # 모델 저장
        model.save(model_file)
        logging.info("새로운 모델을 학습하고 저장했습니다.")
        
        return model, df_scaled, scaler, df_pivot

# 예측 함수
def get_november_prediction(model, df_scaled, scaler, df_pivot):
    look_back = 12  # 예측을 위한 과거 데이터 기간 (12개월)
    
    last_sequence = df_scaled[-look_back:]  # 최근 12개월 데이터
    last_sequence = np.expand_dims(last_sequence, axis=0)  # 3D 형태로 변환
    predicted_next_month = model.predict(last_sequence)

    # 예측 결과 스케일 복원
    predicted_next_month = scaler.inverse_transform(predicted_next_month)

    # 예측 결과 출력
    account_categories = df_pivot.columns.tolist()  # 계정과목 목록
    predicted_data = {}

    # 예측값을 카테고리별로 조정
    for i, account in enumerate(account_categories):
        predicted_value = round(float(predicted_next_month[0, i]), 2)
        
        # 특정 카테고리에 대해 2배 조정
        if account in ['소모품비']:
            adjusted_value = predicted_value * 2  # 2배 증가
        else:
            adjusted_value = predicted_value * 1.5  # 그 외에는 1.5배 증가

        predicted_data[account] = adjusted_value

    return predicted_data

# 실제 11월 데이터 가져오기
def get_actual_november_data():
    today = datetime.now()
    today_str = today.strftime('%Y-%m-%d')
    query_nov = f"""
        SELECT category, SUM(amount) AS Amount
        FROM 사용금액
        WHERE date >= '2024-11-01' AND date <= '{today_str}'
        GROUP BY category
    """
    df_nov = pd.read_sql(query_nov, engine)

    november_usage = {}
    for _, row in df_nov.iterrows():
        november_usage[row['category']] = round(row['Amount'], 2)

    return november_usage
