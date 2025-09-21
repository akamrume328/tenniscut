"""
サンプルデータとテスト用のダミーデータ生成
"""

import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict
import random


def create_sample_reservation_data() -> pd.DataFrame:
    """
    テスト用のサンプル予約データを作成
    H列: 一般申込開始日
    """
    
    today = date.today()
    
    # サンプルデータ
    sample_data = [
        {
            'facility': 'テニスコートA',
            'usage_date': today + timedelta(days=7),
            'time': '10:00-12:00',
            'H': today,  # 今日が一般申込開始日
            'status': 'untouched'
        },
        {
            'facility': 'テニスコートB', 
            'usage_date': today + timedelta(days=10),
            'time': '14:00-16:00',
            'H': today,  # 今日が一般申込開始日
            'status': 'untouched'
        },
        {
            'facility': 'テニスコートC',
            'usage_date': today + timedelta(days=5),
            'time': '09:00-11:00', 
            'H': today - timedelta(days=2),  # 2日前に申込開始済み
            'status': 'untouched'
        },
        {
            'facility': 'テニスコートD',
            'usage_date': today + timedelta(days=8),
            'time': '16:00-18:00',
            'H': today - timedelta(days=1),  # 昨日申込開始済み
            'status': 'untouched'
        },
        {
            'facility': 'テニスコートE',
            'usage_date': today + timedelta(days=3),
            'time': '13:00-15:00',
            'H': today + timedelta(days=1),  # 明日申込開始予定
            'status': 'untouched'
        },
        {
            'facility': 'テニスコートF',
            'usage_date': today + timedelta(days=6),
            'time': '11:00-13:00',
            'H': today,  # 今日が一般申込開始日（既に処理済み）
            'status': 'completed'
        }
    ]
    
    return pd.DataFrame(sample_data)


def create_extended_sample_data(num_courts: int = 20) -> pd.DataFrame:
    """
    拡張サンプルデータを作成（より多くのテストケース）
    """
    
    today = date.today()
    facilities = [f'テニスコート{chr(65 + i)}' for i in range(num_courts)]  # A, B, C, ...
    times = ['09:00-11:00', '11:00-13:00', '13:00-15:00', '15:00-17:00', '17:00-19:00']
    statuses = ['untouched', 'in_progress', 'completed']
    
    data = []
    
    for i in range(num_courts):
        # 一般申込開始日のバリエーション
        start_date_offset = random.randint(-5, 5)  # 5日前から5日後
        general_start_date = today + timedelta(days=start_date_offset)
        
        # 利用日（一般申込開始日より後）
        usage_date_offset = random.randint(7, 30)  # 1週間後から1ヶ月後
        usage_date = general_start_date + timedelta(days=usage_date_offset)
        
        record = {
            'facility': facilities[i % len(facilities)],
            'usage_date': usage_date,
            'time': random.choice(times),
            'H': general_start_date,
            'status': random.choice(statuses) if start_date_offset < 0 else 'untouched'
        }
        
        data.append(record)
    
    return pd.DataFrame(data)


def print_sample_data_summary(df: pd.DataFrame):
    """サンプルデータの概要を表示"""
    
    today = date.today()
    print("=" * 50)
    print("サンプル予約データ概要")
    print("=" * 50)
    
    print(f"本日: {today}")
    print(f"総予約数: {len(df)}")
    print()
    
    # 今日が一般申込開始日の予約
    today_start = df[pd.to_datetime(df['H']).dt.date == today]
    print(f"本日一般申込開始: {len(today_start)}件")
    for _, row in today_start.iterrows():
        print(f"  - {row['facility']} | {row['usage_date']} {row['time']} | ステータス: {row['status']}")
    print()
    
    # 一般申込期間中（開始済み）の予約
    ongoing = df[pd.to_datetime(df['H']).dt.date < today]
    print(f"一般申込期間中: {len(ongoing)}件")
    for _, row in ongoing.iterrows():
        print(f"  - {row['facility']} | 開始日: {row['H']} | ステータス: {row['status']}")
    print()
    
    # 未来の一般申込開始予定
    future = df[pd.to_datetime(df['H']).dt.date > today]
    print(f"申込開始予定: {len(future)}件")
    for _, row in future.iterrows():
        print(f"  - {row['facility']} | 開始予定日: {row['H']}")
    
    print("=" * 50)


if __name__ == "__main__":
    # サンプルデータのテスト
    print("基本サンプルデータの作成...")
    basic_data = create_sample_reservation_data()
    print_sample_data_summary(basic_data)
    
    print("\n拡張サンプルデータの作成...")
    extended_data = create_extended_sample_data(10)
    print_sample_data_summary(extended_data)