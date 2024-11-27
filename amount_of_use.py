import pymysql
import pandas as pd
from sqlalchemy import create_engine
import json

def get_monthly_data():
    # MySQL 연결 정보
    host = "shcrm.ddns.net"
    port = 5001
    database = "shcrm"
    user = "root"
    password = "9868"

    # SQLAlchemy 연결 URL
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")

    # SQL 쿼리로 사용금액 데이터 가져오기 (전체 데이터를 가져옴)
    query = """
        SELECT date, category, amount, payment_method
        FROM 사용금액
        ORDER BY date
    """
    df = pd.read_sql(query, engine)

    # 날짜 컬럼을 datetime 형식으로 변환
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # 월별로 그룹화하여 카테고리별 금액 총합과 결제 방법별 금액 총합 계산
    df['month'] = df['date'].dt.to_period('M')  # 월별로 그룹화

    # 카테고리별 월간 총합 계산
    df_category = df.groupby(['month', 'category'])['amount'].sum().reset_index()
    df_category_pivot = df_category.pivot_table(index='month', columns='category', values='amount', aggfunc='sum').fillna(0)

    # 결제 방법별 월간 총합 계산
    df_payment = df.groupby(['month', 'payment_method'])['amount'].sum().reset_index()
    df_payment_pivot = df_payment.pivot_table(index='month', columns='payment_method', values='amount', aggfunc='sum').fillna(0)

    # 월별 전체 총합 계산
    df_category_pivot['total'] = df_category_pivot.sum(axis=1)

    # 결과를 JSON 형식으로 변환
    result_data = {
        "monthly_totals": []
    }

    # 월별로 데이터 추가
    for index in df_category_pivot.index:
        monthly_data = {
            "month": str(index),
            "categories": df_category_pivot.loc[index].drop('total').astype(int).to_dict(),  # 카테고리별 금액 합계
            "payment_method": df_payment_pivot.loc[index].astype(int).to_dict(),  # 결제 방법별 금액 합계
            "total": int(df_category_pivot.loc[index]['total'])  # 월간 전체 합계
        }
        result_data["monthly_totals"].append(monthly_data)

    return result_data
