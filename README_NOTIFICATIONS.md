# テニス予約LINE通知システム

## 概要

このシステムは、テニス予約の「一般申込開始日」を監視し、適切なタイミングでLINE通知を送信する機能を提供します。

## 主な機能

### 🚨 本日対応必須アラート
- 今日が「一般申込開始日（H列）」となる予約を検出
- 【一般申込開始！】本日から申込可能（早い者勝ち）」のような強調メッセージで通知
- 緊急度の高い通知として最優先で表示

### 📅 毎日通知
- 一般申込期間中（開始日～利用日まで）の予約を継続的に通知
- 【一般申込受付中】セクションで整理して表示

### 📱 LINE通知連携
- LINE Notify API を使用した自動通知
- テスト用のモック機能も提供

## インストール

### 必要な依存関係
```bash
pip install pandas requests schedule
```

### ファイル構成
```
notifications.py          # メイン通知システム
line_integration.py      # LINE API 連携
sample_data.py          # テストデータ生成
test_notifications.py   # テスト実行
integrated_demo.py      # 統合デモ
```

## 使用方法

### 1. 基本的な使用例

```python
from notifications import ReservationNotificationManager
from line_integration import create_line_service
import pandas as pd

# 通知マネージャーの初期化
manager = ReservationNotificationManager()

# 予約データの準備（H列に一般申込開始日を含む）
reservations_df = pd.DataFrame({
    'facility': ['テニスコートA'],
    'usage_date': ['2024-09-28'],
    'time': ['10:00-12:00'],
    'H': ['2024-09-21'],  # 一般申込開始日
    'status': ['untouched']
})

# 通知チェック実行
notifications = manager.checkDailyNotifications(reservations_df)

# LINE通知送信
line_service = create_line_service(use_mock=True)
result = line_service.send_reservation_alert(notifications)
```

### 2. テスト実行

```bash
# 基本テスト
python test_notifications.py

# 統合デモ（自動テスト）
python integrated_demo.py test

# インタラクティブデモ
python integrated_demo.py demo
```

### 3. 定期実行の設定

```python
from integrated_demo import TennisReservationNotificationSystem

# システムの初期化
system = TennisReservationNotificationSystem(use_mock_line=False, 
                                           line_access_token="YOUR_TOKEN")

# 定期通知設定
system.setup_scheduled_notifications()

# スケジューラー開始
system.run_scheduler()
```

## 設定

### LINE Notify トークンの取得

1. [LINE Notify](https://notify-api.line.me/) にアクセス
2. 「マイページ」からトークンを発行
3. 発行されたトークンを設定

```python
# 実際のLINE通知を使用する場合
line_service = create_line_service(use_mock=False, 
                                 access_token="YOUR_LINE_NOTIFY_TOKEN")
```

### データ形式

予約データは以下の列を含むPandas DataFrameで提供してください：

- `facility`: 施設名（例: "テニスコートA"）
- `usage_date`: 利用日
- `time`: 利用時間（例: "10:00-12:00"）
- `H`: 一般申込開始日（重要）
- `status`: 予約状態（"untouched", "in_progress", "completed"）

## API リファレンス

### ReservationNotificationManager

#### checkDailyNotifications(reservations_df)
毎日の通知をチェックし、分類する。

**パラメータ:**
- `reservations_df`: 予約データフレーム

**戻り値:**
```python
{
    'todayActionBody': [],      # 本日対応必須
    'generalApplication': [],   # 一般申込（毎日通知）
    'other': []                # その他の通知
}
```

### LineNotificationService

#### send_notification(message, image_path=None)
LINE通知を送信する。

**パラメータ:**
- `message`: 送信するメッセージ
- `image_path`: 添付画像のパス（オプション）

## 通知メッセージ例

```
🚨【本日対応必須】🚨
【一般申込開始！】本日から申込可能（早い者勝ち）
🎾 テニスコートA | 2024-09-28 10:00-12:00
🎾 テニスコートB | 2024-09-30 14:00-16:00

📅【毎日通知】
【一般申込受付中】
🎾 テニスコートC | 2024-09-26 09:00-11:00
🎾 テニスコートD | 2024-09-27 16:00-18:00

⏰ 送信時刻: 2024-09-21 08:00
```

## デプロイメント

### 1. サーバー環境での実行

```bash
# 依存関係のインストール
pip install -r requirements.txt

# システム起動
python integrated_demo.py schedule
```

### 2. cron での定期実行

```bash
# crontab -e で編集
# 毎日8時に実行
0 8 * * * /usr/bin/python3 /path/to/tennis_notifications.py
```

### 3. Docker での実行

```dockerfile
FROM python:3.12
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "integrated_demo.py", "schedule"]
```

## トラブルシューティング

### よくある問題

1. **LINE通知が送信されない**
   - アクセストークンが正しく設定されているか確認
   - LINE Notify の設定を確認

2. **日付の比較エラー**
   - H列のデータ形式が正しいか確認（YYYY-MM-DD形式）
   - データにNaN値が含まれていないか確認

3. **通知が重複する**
   - 同じ時間に複数回実行されていないか確認
   - ステータス列の管理を確認

### ログ出力

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 開発・カスタマイズ

### 新しい通知タイプの追加

```python
class CustomNotificationManager(ReservationNotificationManager):
    def checkDailyNotifications(self, reservations_df):
        notifications = super().checkDailyNotifications(reservations_df)
        
        # カスタム通知ロジックを追加
        custom_notifications = self._check_custom_conditions(reservations_df)
        notifications['custom'] = custom_notifications
        
        return notifications
```

### メッセージフォーマットのカスタマイズ

```python
def custom_format_message(reservation):
    return f"🎾 {reservation['facility']} - {reservation['date']} {reservation['time']}"
```

## ライセンス

このプロジェクトは MIT ライセンスのもとで公開されています。