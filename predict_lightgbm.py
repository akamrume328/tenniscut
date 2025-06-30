import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
from datetime import datetime
import argparse
from typing import Optional

class LightGBMPredictor:
    """学習済みのLightGBMモデルを使って予測を行うクラス"""

    def __init__(self, output_dir: Path):
        """
        コンストラクタ

        Args:
            output_dir (Path): 予測結果を保存するディレクトリ
        """
        self.predictions_output_dir = output_dir
        self.predictions_output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        # IDからラベル名への変換マップ
        self.phase_labels_map = {
            0: "point_interval", 1: "rally", 2: "serve_front_deuce", 3: "serve_front_ad",
            4: "serve_back_deuce", 5: "serve_back_ad", 6: "changeover"
        }

    def load_model(self, model_path: Path) -> bool:
        """モデルファイルを読み込む"""
        if not model_path.exists():
            print(f"❌ モデルファイルが見つかりません: {model_path}")
            return False
        
        print(f"モデルを読み込んでいます: {model_path}")
        self.model = joblib.load(model_path)
        return True

    def run_prediction_for_file(self, model_path: Path, feature_csv_path: Path) -> Optional[Path]:
        """
        単一の特徴量CSVファイルに対して予測を実行し、結果をCSVとして保存する。
        """
        # 1. モデルの読み込み
        if not self.load_model(model_path):
            return None

        # 2. 予測用データの読み込み
        if not feature_csv_path.exists():
            print(f"❌ 入力CSVファイルが見つかりません: {feature_csv_path}")
            return None

        print(f"予測用データを読み込んでいます: {feature_csv_path}")
        df_inference = pd.read_csv(feature_csv_path)

        # 3. 特徴量の整合性を確認
        model_features = self.model.feature_name_
        missing_features = [f for f in model_features if f not in df_inference.columns]
        if missing_features:
            print(f"❌ 入力データに必須の特徴量が不足しています: {missing_features}")
            return None

        X_inference = df_inference[model_features]
        print(f"{len(model_features)}個の特徴量を使って予測を実行します。")

        # 4. 予測の実行
        print("予測を開始します...")
        predicted_probas = self.model.predict_proba(X_inference)
        predicted_ids = np.argmax(predicted_probas, axis=1)
        prediction_confidence = np.max(predicted_probas, axis=1)
        print("予測が完了しました。")

        # 5. 結果の整形と保存
        results_df = pd.DataFrame()
        id_columns = ['video_name', 'frame_number', 'original_frame_number']
        for col in id_columns:
            if col in df_inference.columns:
                results_df[col] = df_inference[col]

        results_df['predicted_label_id'] = predicted_ids
        results_df['predicted_phase'] = results_df['predicted_label_id'].map(self.phase_labels_map)
        results_df['prediction_confidence'] = prediction_confidence
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"predictions_{feature_csv_path.stem}_{timestamp}.csv"
        output_path = self.predictions_output_dir / output_filename

        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"LGBM予測結果を保存しました: {output_path}")
        return output_path

# このスクリプトを直接実行した場合の動作（コマンドラインツールとして）
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="学習済みLightGBMモデルによる推論スクリプト")
    parser.add_argument('--model_path', type=str, required=True, help="学習済みモデルのパス (.pkl)")
    parser.add_argument('--input_csv', type=str, required=True, help="予測対象のデータCSV（特徴量抽出済み）")
    parser.add_argument('--output_dir', type=str, default="predictions_output", help="(オプション) 予測結果を保存するディレクトリ")
    
    args = parser.parse_args()
    
    output_dir_path = Path(args.output_dir)
    predictor = LightGBMPredictor(output_dir=output_dir_path)
    predictor.run_prediction_for_file(
        model_path=Path(args.model_path),
        feature_csv_path=Path(args.input_csv)
    )