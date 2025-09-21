"""
LINE通知システム - テニス予約管理
"""

import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReservationNotificationManager:
    """予約通知管理クラス"""
    
    def __init__(self):
        self.today = date.today()
        
    def checkDailyNotifications(self, reservations_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        毎日の通知をチェックし、分類する
        
        Args:
            reservations_df: 予約データフレーム（H列に一般申込開始日を含む）
            
        Returns:
            分類された通知データ
        """
        logger.info(f"Daily notifications check started for {self.today}")
        
        # データフレームのコピーを作成
        df = reservations_df.copy()
        
        # 日付列を確実にdatetime型に変換
        if 'H' in df.columns:  # H列：一般申込開始日
            df['H'] = pd.to_datetime(df['H'], errors='coerce')
        
        # 通知結果の初期化
        notifications = {
            'todayActionBody': [],      # 本日対応必須
            'generalApplication': [],   # 一般申込（毎日通知）
            'other': []                # その他の通知
        }
        
        # 今日が一般申込開始日の予約を抽出
        generalApplicationToday = self._extractGeneralApplicationToday(df)
        
        # 一般申込期間中の他の予約を抽出
        generalApplicationOngoing = self._extractGeneralApplicationOngoing(df)
        
        # 本日対応必須セクションに一般申込開始日の予約を追加
        if generalApplicationToday:
            urgent_section = self._formatGeneralApplicationTodaySection(generalApplicationToday)
            notifications['todayActionBody'].extend(urgent_section)
            
        # 毎日通知セクションに進行中の一般申込を追加
        if generalApplicationOngoing:
            ongoing_section = self._formatSection(generalApplicationOngoing, '一般申込受付中')
            notifications['generalApplication'].extend(ongoing_section)
            
        logger.info(f"Notifications classified: {len(notifications['todayActionBody'])} urgent, "
                   f"{len(notifications['generalApplication'])} ongoing")
        
        return notifications
    
    def _extractGeneralApplicationToday(self, df: pd.DataFrame) -> List[Dict]:
        """今日が一般申込開始日の予約を抽出"""
        if 'H' not in df.columns:
            return []
            
        # 日付が有効な行のみを対象とする
        valid_date_mask = df['H'].notna()
        if not valid_date_mask.any():
            return []
            
        # 今日が一般申込開始日の行を抽出
        try:
            today_mask = df.loc[valid_date_mask, 'H'].dt.date == self.today
        except (AttributeError, TypeError):
            # 日付変換に失敗した場合は空のマスクを返す
            today_mask = pd.Series([False] * len(df))
            
        untouched_mask = df.get('status', 'untouched') == 'untouched'  # 未着手のもの
        
        final_mask = valid_date_mask & today_mask & untouched_mask
        filtered_df = df[final_mask]
        
        return self._convertToNotificationFormat(filtered_df)
    
    def _extractGeneralApplicationOngoing(self, df: pd.DataFrame) -> List[Dict]:
        """一般申込期間中（開始日以降、利用日まで）の予約を抽出"""
        if 'H' not in df.columns:
            return []
            
        # 日付が有効な行のみを対象とする
        valid_date_mask = df['H'].notna()
        if not valid_date_mask.any():
            return []
            
        # 一般申込開始日が過ぎて、今日が開始日ではない予約
        try:
            start_passed_mask = df.loc[valid_date_mask, 'H'].dt.date < self.today
        except (AttributeError, TypeError):
            # 日付変換に失敗した場合は空のマスクを返す
            start_passed_mask = pd.Series([False] * len(df))
            
        untouched_mask = df.get('status', 'untouched') == 'untouched'
        
        # 利用日がまだ過ぎていない予約（利用日列があると仮定）
        if 'usage_date' in df.columns:
            df['usage_date'] = pd.to_datetime(df['usage_date'], errors='coerce')
            usage_valid_mask = df['usage_date'].notna()
            try:
                usage_future_mask = df.loc[usage_valid_mask, 'usage_date'].dt.date >= self.today
            except (AttributeError, TypeError):
                usage_future_mask = pd.Series([True] * len(df))
        else:
            usage_future_mask = pd.Series([True] * len(df))  # 利用日列がない場合は全てTrue
        
        final_mask = valid_date_mask & start_passed_mask & untouched_mask & usage_future_mask
        filtered_df = df[final_mask]
        
        return self._convertToNotificationFormat(filtered_df)
    
    def _convertToNotificationFormat(self, df: pd.DataFrame) -> List[Dict]:
        """データフレームを通知フォーマットに変換"""
        notifications = []
        
        for _, row in df.iterrows():
            notification = {
                'facility': row.get('facility', '未設定'),
                'date': row.get('usage_date', '未設定'),
                'time': row.get('time', '未設定'), 
                'general_start_date': row.get('H', '未設定'),
                'status': row.get('status', 'untouched'),
                'raw_data': row.to_dict()
            }
            notifications.append(notification)
            
        return notifications
    
    def _formatGeneralApplicationTodaySection(self, reservations: List[Dict]) -> List[str]:
        """本日一般申込開始の通知セクションをフォーマット"""
        if not reservations:
            return []
            
        formatted_messages = []
        
        # 強調タイトル
        title = "【一般申込開始！】本日から申込可能（早い者勝ち）"
        formatted_messages.append(title)
        
        # 各予約の詳細
        for reservation in reservations:
            message = self._formatReservationMessage(reservation)
            formatted_messages.append(message)
            
        return formatted_messages
    
    def _formatSection(self, reservations: List[Dict], section_title: str) -> List[str]:
        """通常の通知セクションをフォーマット"""
        if not reservations:
            return []
            
        formatted_messages = []
        
        if section_title:
            formatted_messages.append(f"【{section_title}】")
            
        for reservation in reservations:
            message = self._formatReservationMessage(reservation)
            formatted_messages.append(message)
            
        return formatted_messages
    
    def _formatReservationMessage(self, reservation: Dict) -> str:
        """個別の予約メッセージをフォーマット"""
        facility = reservation.get('facility', '未設定')
        date = reservation.get('date', '未設定')
        time = reservation.get('time', '未設定')
        
        # 日付のフォーマット調整
        if isinstance(date, (pd.Timestamp, datetime)):
            date = date.strftime('%Y-%m-%d')
        
        return f"🎾 {facility} | {date} {time}"
    
    def generateLineNotificationMessage(self, notifications: Dict[str, List]) -> str:
        """LINE通知用のメッセージを生成"""
        message_parts = []
        
        # 最重要アラート（本日対応必須）
        if notifications['todayActionBody']:
            message_parts.append("🚨【本日対応必須】🚨")
            message_parts.extend(notifications['todayActionBody'])
            message_parts.append("")  # 空行
        
        # 毎日通知
        if notifications['generalApplication']:
            message_parts.append("📅【毎日通知】")
            message_parts.extend(notifications['generalApplication'])
            message_parts.append("")  # 空行
            
        # その他の通知
        if notifications['other']:
            message_parts.append("ℹ️【その他】")
            message_parts.extend(notifications['other'])
        
        return "\n".join(message_parts)


def formatSection(reservations: List[Dict], prefix: str = '') -> List[str]:
    """
    後方互換性のための関数
    """
    manager = ReservationNotificationManager()
    return manager._formatSection(reservations, prefix)