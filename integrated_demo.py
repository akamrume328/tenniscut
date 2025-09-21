"""
テニス予約通知システム - 統合デモ
LINE通知システムの完全な実装例
"""

import sys
import os
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    print("⚠️ schedule モジュールが見つかりません。定期実行機能は無効です。")

import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from notifications import ReservationNotificationManager
from line_integration import create_line_service
from sample_data import create_sample_reservation_data, create_extended_sample_data


class TennisReservationNotificationSystem:
    """テニス予約通知システムのメインクラス"""
    
    def __init__(self, use_mock_line=True, line_access_token=None):
        """
        初期化
        
        Args:
            use_mock_line: モックLINE通知を使用するかどうか
            line_access_token: 実際のLINE Notify アクセストークン
        """
        self.notification_manager = ReservationNotificationManager()
        self.line_service = create_line_service(use_mock_line, line_access_token)
        self.last_notification_time = None
        
    def load_reservation_data(self, data_source="sample") -> pd.DataFrame:
        """
        予約データを読み込み
        
        Args:
            data_source: データソース ("sample", "csv", "database" など)
            
        Returns:
            予約データフレーム
        """
        if data_source == "sample":
            return create_sample_reservation_data()
        elif data_source == "extended":
            return create_extended_sample_data(20)
        elif data_source == "csv":
            # CSVファイルから読み込み（実装例）
            # return pd.read_csv("reservations.csv")
            pass
        elif data_source == "database":
            # データベースから読み込み（実装例）
            # return self._load_from_database()
            pass
        else:
            raise ValueError(f"Unknown data source: {data_source}")
    
    def run_daily_notification_check(self, data_source="sample"):
        """
        毎日の通知チェックを実行
        
        Args:
            data_source: データソース
        """
        print(f"\n🔔 毎日通知チェック実行中... ({datetime.now()})")
        
        try:
            # 予約データの読み込み
            reservations_df = self.load_reservation_data(data_source)
            print(f"📊 予約データ読み込み完了: {len(reservations_df)}件")
            
            # 通知チェック実行
            notifications = self.notification_manager.checkDailyNotifications(reservations_df)
            
            # 通知が必要かチェック
            total_notifications = sum(len(v) for v in notifications.values())
            
            if total_notifications > 0:
                print(f"📢 {total_notifications}件の通知が発生")
                
                # LINE通知送信
                result = self.line_service.send_reservation_alert(notifications)
                
                if result["status"] == "success":
                    print("✅ LINE通知送信成功")
                    self.last_notification_time = datetime.now()
                else:
                    print(f"❌ LINE通知送信失敗: {result}")
            else:
                print("ℹ️ 今日は通知対象の予約がありません")
                
        except Exception as e:
            print(f"❌ 通知チェック中にエラーが発生: {e}")
            import traceback
            traceback.print_exc()
    
    def run_test_notification(self, data_source="sample"):
        """
        テスト通知を実行
        
        Args:
            data_source: データソース
        """
        print(f"\n🧪 テスト通知実行中...")
        self.run_daily_notification_check(data_source)
    
    def setup_scheduled_notifications(self):
        """
        定期通知のスケジュール設定
        """
        if not SCHEDULE_AVAILABLE:
            print("❌ scheduleモジュールが利用できません")
            return
        
        # 毎日朝8時に通知チェック
        schedule.every().day.at("08:00").do(self.run_daily_notification_check)
        
        # 平日の夕方18時にも追加チェック（オプション）
        schedule.every().monday.at("18:00").do(self.run_daily_notification_check)
        schedule.every().tuesday.at("18:00").do(self.run_daily_notification_check)
        schedule.every().wednesday.at("18:00").do(self.run_daily_notification_check)
        schedule.every().thursday.at("18:00").do(self.run_daily_notification_check)
        schedule.every().friday.at("18:00").do(self.run_daily_notification_check)
        
        print("⏰ 定期通知スケジュール設定完了")
        print("   - 毎日 08:00: 毎日通知チェック")
        print("   - 平日 18:00: 追加通知チェック")
    
    def run_scheduler(self):
        """
        スケジューラーの実行（デーモンモード）
        """
        if not SCHEDULE_AVAILABLE:
            print("❌ scheduleモジュールが利用できません")
            return
            
        print("🚀 通知スケジューラー開始...")
        print("   Ctrl+C で停止")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # 1分間隔でチェック
        except KeyboardInterrupt:
            print("\n⏹️ スケジューラー停止")
    
    def run_interactive_demo(self):
        """
        インタラクティブデモの実行
        """
        print("\n🎮 テニス予約通知システム - インタラクティブデモ")
        print("=" * 60)
        
        while True:
            print("\n📋 メニュー:")
            print("1. 基本データでテスト通知")
            print("2. 拡張データでテスト通知")
            print("3. LINE接続テスト")
            print("4. 通知履歴表示")
            print("5. 定期通知設定")
            print("6. 終了")
            
            choice = input("\n選択してください (1-6): ").strip()
            
            if choice == "1":
                self.run_test_notification("sample")
            elif choice == "2":
                self.run_test_notification("extended")
            elif choice == "3":
                result = self.line_service.test_connection()
                print(f"📱 接続テスト結果: {result}")
            elif choice == "4":
                if hasattr(self.line_service, 'get_sent_messages'):
                    messages = self.line_service.get_sent_messages()
                    print(f"📋 送信履歴: {len(messages)}件")
                    for i, msg in enumerate(messages[-5:], 1):  # 最新5件を表示
                        print(f"   {i}. {msg['timestamp']}")
                else:
                    print("📋 履歴機能はモックサービスでのみ利用可能です")
            elif choice == "5":
                self.setup_scheduled_notifications()
                print("⚠️ 実際の定期実行を開始するには run_scheduler() を呼び出してください")
            elif choice == "6":
                print("👋 デモを終了します")
                break
            else:
                print("❌ 無効な選択です")


def main():
    """メイン実行関数"""
    
    print("🎾 テニス予約通知システム - 統合デモ")
    print("=" * 60)
    
    # システムの初期化
    system = TennisReservationNotificationSystem(use_mock_line=True)
    
    # コマンドライン引数のチェック
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "test":
            print("🧪 自動テストモード")
            system.run_test_notification("sample")
            system.run_test_notification("extended")
            
        elif mode == "schedule":
            print("⏰ スケジュールモード")
            system.setup_scheduled_notifications()
            system.run_scheduler()
            
        elif mode == "demo":
            print("🎮 デモモード")
            system.run_interactive_demo()
            
        else:
            print(f"❌ 不明なモード: {mode}")
            print("利用可能なモード: test, schedule, demo")
            return 1
    else:
        # デフォルトはインタラクティブデモ
        system.run_interactive_demo()
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 プログラムを終了します")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)