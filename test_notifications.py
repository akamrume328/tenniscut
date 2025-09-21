"""
通知システムのテストとデモ
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from notifications import ReservationNotificationManager
from sample_data import create_sample_reservation_data, create_extended_sample_data, print_sample_data_summary


def test_daily_notifications():
    """毎日通知機能のテスト"""
    
    print("🚀 テニス予約通知システム - デモ実行")
    print("=" * 60)
    
    # サンプルデータの作成
    print("📊 サンプルデータを作成中...")
    df = create_sample_reservation_data()
    print_sample_data_summary(df)
    
    # 通知マネージャーの初期化
    print("\n📢 通知システムを初期化中...")
    notification_manager = ReservationNotificationManager()
    
    # 毎日通知のチェック実行
    print("\n🔍 毎日通知をチェック中...")
    notifications = notification_manager.checkDailyNotifications(df)
    
    # 結果の表示
    print("\n📋 通知結果:")
    print("-" * 40)
    
    print(f"🚨 本日対応必須: {len(notifications['todayActionBody'])}件")
    if notifications['todayActionBody']:
        for msg in notifications['todayActionBody']:
            print(f"   {msg}")
    
    print(f"\n📅 一般申込（毎日通知）: {len(notifications['generalApplication'])}件") 
    if notifications['generalApplication']:
        for msg in notifications['generalApplication']:
            print(f"   {msg}")
    
    print(f"\nℹ️ その他: {len(notifications['other'])}件")
    if notifications['other']:
        for msg in notifications['other']:
            print(f"   {msg}")
    
    # LINE通知メッセージの生成
    print("\n📱 LINE通知メッセージ:")
    print("-" * 40)
    line_message = notification_manager.generateLineNotificationMessage(notifications)
    print(line_message)
    
    return notifications


def test_extended_notifications():
    """拡張データでの通知テスト"""
    
    print("\n\n🔄 拡張データでのテスト")
    print("=" * 60)
    
    # 拡張サンプルデータの作成
    df = create_extended_sample_data(15)
    print_sample_data_summary(df)
    
    # 通知マネージャーの初期化
    notification_manager = ReservationNotificationManager()
    
    # 毎日通知のチェック実行
    notifications = notification_manager.checkDailyNotifications(df)
    
    # 結果の表示
    print("\n📋 通知結果（拡張データ）:")
    print("-" * 40)
    
    print(f"🚨 本日対応必須: {len(notifications['todayActionBody'])}件")
    print(f"📅 一般申込（毎日通知）: {len(notifications['generalApplication'])}件")
    print(f"ℹ️ その他: {len(notifications['other'])}件")
    
    # LINE通知メッセージの生成と表示
    print("\n📱 LINE通知メッセージ（拡張版）:")
    print("-" * 40)
    line_message = notification_manager.generateLineNotificationMessage(notifications)
    print(line_message)
    
    return notifications


def test_edge_cases():
    """エッジケースのテスト"""
    
    print("\n\n⚠️ エッジケースのテスト")
    print("=" * 60)
    
    import pandas as pd
    from datetime import date
    
    # 空のデータフレーム
    print("1. 空のデータフレームのテスト")
    empty_df = pd.DataFrame()
    manager = ReservationNotificationManager()
    result = manager.checkDailyNotifications(empty_df)
    print(f"   結果: {sum(len(v) for v in result.values())}件の通知")
    
    # H列がないデータフレーム
    print("\n2. H列がないデータフレームのテスト")
    no_h_column_df = pd.DataFrame({
        'facility': ['テストコート'],
        'usage_date': [date.today()],
        'time': ['10:00-12:00']
    })
    result = manager.checkDailyNotifications(no_h_column_df)
    print(f"   結果: {sum(len(v) for v in result.values())}件の通知")
    
    # 無効な日付データのテスト
    print("\n3. 無効な日付データのテスト")
    invalid_date_df = pd.DataFrame({
        'facility': ['テストコート'],
        'usage_date': [date.today()],
        'time': ['10:00-12:00'],
        'H': ['invalid_date'],
        'status': ['untouched']
    })
    result = manager.checkDailyNotifications(invalid_date_df)
    print(f"   結果: {sum(len(v) for v in result.values())}件の通知")
    
    print("\n✅ エッジケーステスト完了")


def main():
    """メイン実行関数"""
    
    try:
        # 基本的な通知テスト
        basic_notifications = test_daily_notifications()
        
        # 拡張データでのテスト
        extended_notifications = test_extended_notifications()
        
        # エッジケースのテスト
        test_edge_cases()
        
        print("\n\n🎉 全てのテストが正常に完了しました！")
        print("=" * 60)
        
        # まとめ
        print("\n📊 テスト結果まとめ:")
        print(f"基本テスト - 本日対応必須: {len(basic_notifications['todayActionBody'])}件")
        print(f"基本テスト - 一般申込継続: {len(basic_notifications['generalApplication'])}件")
        print(f"拡張テスト - 本日対応必須: {len(extended_notifications['todayActionBody'])}件")
        print(f"拡張テスト - 一般申込継続: {len(extended_notifications['generalApplication'])}件")
        
    except Exception as e:
        print(f"\n❌ テスト実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)