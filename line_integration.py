"""
LINE通知連携モジュール
"""

import requests
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime


class LineNotificationService:
    """LINE通知サービスクラス"""
    
    def __init__(self, access_token: Optional[str] = None):
        """
        初期化
        
        Args:
            access_token: LINE Notify アクセストークン
        """
        self.access_token = access_token
        self.line_notify_url = "https://notify-api.line.me/api/notify"
        self.logger = logging.getLogger(__name__)
    
    def send_notification(self, message: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        LINE通知を送信
        
        Args:
            message: 送信するメッセージ
            image_path: 添付する画像のパス（オプション）
            
        Returns:
            送信結果
        """
        if not self.access_token:
            self.logger.warning("LINE Notify access token is not set")
            return {"status": "error", "message": "Access token not configured"}
        
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        
        data = {
            "message": message
        }
        
        files = None
        if image_path:
            try:
                files = {"imageFile": open(image_path, "rb")}
            except FileNotFoundError:
                self.logger.warning(f"Image file not found: {image_path}")
        
        try:
            response = requests.post(
                self.line_notify_url,
                headers=headers,
                data=data,
                files=files
            )
            
            if files:
                files["imageFile"].close()
            
            if response.status_code == 200:
                self.logger.info("LINE notification sent successfully")
                return {"status": "success", "response": response.json()}
            else:
                self.logger.error(f"Failed to send LINE notification: {response.status_code}")
                return {
                    "status": "error", 
                    "message": f"HTTP {response.status_code}",
                    "response": response.text
                }
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error while sending LINE notification: {e}")
            return {"status": "error", "message": str(e)}
    
    def send_reservation_alert(self, notifications: Dict) -> Dict[str, Any]:
        """
        予約アラート専用の通知送信
        
        Args:
            notifications: 通知データ
            
        Returns:
            送信結果
        """
        from notifications import ReservationNotificationManager
        
        manager = ReservationNotificationManager()
        message = manager.generateLineNotificationMessage(notifications)
        
        # タイムスタンプを追加
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        final_message = f"{message}\n\n⏰ 送信時刻: {timestamp}"
        
        return self.send_notification(final_message)
    
    def test_connection(self) -> Dict[str, Any]:
        """
        LINE Notify との接続をテスト
        
        Returns:
            テスト結果
        """
        test_message = "🧪 テニス予約通知システム - 接続テスト"
        return self.send_notification(test_message)


class MockLineNotificationService(LineNotificationService):
    """
    開発・テスト用のモックLINE通知サービス
    実際にLINEに送信せず、コンソールに出力する
    """
    
    def __init__(self):
        super().__init__(access_token="mock_token")
        self.sent_messages = []  # 送信されたメッセージの履歴
    
    def send_notification(self, message: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        モック通知送信（コンソール出力）
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "="*60)
        print("📱 LINE通知 (Mock)")
        print("="*60)
        print(f"⏰ 送信時刻: {timestamp}")
        if image_path:
            print(f"🖼️ 添付画像: {image_path}")
        print("\n📝 メッセージ:")
        print("-"*40)
        print(message)
        print("="*60)
        
        # 送信履歴に保存
        self.sent_messages.append({
            "timestamp": timestamp,
            "message": message,
            "image_path": image_path
        })
        
        return {
            "status": "success",
            "message": "Mock notification sent successfully",
            "timestamp": timestamp
        }
    
    def get_sent_messages(self) -> list:
        """送信されたメッセージの履歴を取得"""
        return self.sent_messages.copy()
    
    def clear_history(self):
        """送信履歴をクリア"""
        self.sent_messages.clear()


def create_line_service(use_mock: bool = True, access_token: Optional[str] = None) -> LineNotificationService:
    """
    LINE通知サービスのファクトリー関数
    
    Args:
        use_mock: モックサービスを使用するかどうか
        access_token: 実際のLINE Notify アクセストークン
        
    Returns:
        LINE通知サービスインスタンス
    """
    if use_mock:
        return MockLineNotificationService()
    else:
        return LineNotificationService(access_token)


if __name__ == "__main__":
    # モックサービスのテスト
    print("📱 LINE通知サービスのテスト")
    
    # モックサービスの作成
    mock_service = create_line_service(use_mock=True)
    
    # テスト通知の送信
    test_message = """🚨【本日対応必須】🚨
【一般申込開始！】本日から申込可能（早い者勝ち）
🎾 テニスコートA | 2024-09-28 10:00-12:00
🎾 テニスコートB | 2024-09-30 14:00-16:00

📅【毎日通知】
【一般申込受付中】
🎾 テニスコートC | 2024-09-26 09:00-11:00
🎾 テニスコートD | 2024-09-27 16:00-18:00"""
    
    result = mock_service.send_notification(test_message)
    print(f"\n📊 送信結果: {result}")
    
    # 送信履歴の確認
    history = mock_service.get_sent_messages()
    print(f"\n📋 送信履歴: {len(history)}件のメッセージ")