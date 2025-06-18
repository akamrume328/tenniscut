from ultralytics import YOLO
from pathlib import Path  # 追加

def convert_model_to_tensorrt(model_path: str, output_name: str = None, imgsz: int = 640, half: bool = False, int8: bool = False, simplify: bool = False):
    """
    YOLOモデルをTensorRT形式に変換します。

    Args:
        model_path (str): 変換するYOLOモデルのパス (.ptファイルなど)。
        output_name (str, optional): 出力されるTensorRTエンジンの名前 (拡張子なし)。
                                     指定しない場合、元のモデル名から自動生成されます。
        imgsz (int, optional): 推論時の画像サイズ。デフォルトは640。
        half (bool, optional): FP16量子化を使用するかどうか。デフォルトはFalse。
        int8 (bool, optional): INT8量子化を使用するかどうか。デフォルトはFalse。
                               INT8量子化にはキャリブレーションデータセットが必要です。
        simplify (bool, optional): ONNXモデルをエクスポートする際に簡略化するかどうか。デフォルトはFalse。
    """
    try:
        # YOLOモデルをロード
        model = YOLO(model_path)
        print(f"モデルをロードしました: {model_path}")

        # 出力ファイル名の設定
        if output_name is None:
            output_engine_name = Path(model_path).stem
        else:
            output_engine_name = output_name
        
        export_params = {
            'format': 'engine',  # TensorRT形式を指定
            'imgsz': imgsz,
            'half': half,
            'int8': int8,
            'simplify': simplify,
            # 'device': 0,  # GPUデバイスID (必要に応じて指定)
        }

        print(f"TensorRTへのエクスポートを開始します...")
        print(f"パラメータ: {export_params}")

        # モデルをエクスポート
        # export()メソッドは、指定されたフォーマットでモデルをエクスポートし、
        # エクスポートされたモデルのパスを返します。
        # TensorRTの場合、通常は元のモデルと同じディレクトリに .engine ファイルが作成されます。
        exported_model_path = model.export(**export_params)

        print(f"モデルのエクスポートが完了しました。")
        print(f"TensorRTエンジンが保存されました: {exported_model_path}")

    except Exception as e:
        print(f"モデル変換中にエラーが発生しました: {e}")
        print("考えられる原因と対策:")
        print("- TensorRTおよび関連ライブラリ (CUDA, cuDNN) が正しくインストールされているか確認してください。")
        print("- GPUドライバが最新であるか確認してください。")
        print("- INT8量子化を使用する場合、キャリブレーションデータセットが適切に設定されているか確認してください（ultralyticsライブラリのドキュメント参照）。")
        print("- モデルパスが正しいか確認してください。")
        print("- 十分なGPUメモリがあるか確認してください。")

if __name__ == "__main__":
    # --- 設定項目 ---
    # 変換するYOLOモデルのパス (例: "path/to/your/model.pt")
    # best_5_31.pt は balltracking.py の main 関数で指定されているモデルパスです。
    # 環境に合わせて変更してください。
    DEFAULT_MODEL_PATH = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennis_ball_tracking/models/yolo_model/best_5_31.pt"
    
    # 出力するTensorRTエンジンの名前 (拡張子 .engine は自動で付与されます)
    # Noneの場合、元のモデル名が使用されます。
    OUTPUT_ENGINE_NAME = None # 例: "my_yolov8_tensorrt_engine"

    # 推論時の画像サイズ
    IMAGE_SIZE = 1920 # 例: 640, 1280

    # FP16量子化を使用するか (True/False)
    USE_FP16 = True # TensorRTではFP16が一般的に推奨されます

    # INT8量子化を使用するか (True/False) - 注意: INT8にはキャリブレーションが必要です
    USE_INT8 = False

    # ONNXモデルの簡略化 (True/False)
    SIMPLIFY_ONNX = False
    # --- 設定項目ここまで ---

    print("TensorRTモデル変換スクリプト")
    print("--------------------------")
    
    model_to_convert = input(f"変換するモデルのパスを入力してください (デフォルト: {DEFAULT_MODEL_PATH}): ")
    if not model_to_convert:
        model_to_convert = DEFAULT_MODEL_PATH

    imgsz_input = input(f"画像サイズを入力してください (デフォルト: {IMAGE_SIZE}): ")
    try:
        imgsz_to_use = int(imgsz_input) if imgsz_input else IMAGE_SIZE
    except ValueError:
        print(f"無効な画像サイズです。デフォルトの{IMAGE_SIZE}を使用します。")
        imgsz_to_use = IMAGE_SIZE

    use_fp16_input = input(f"FP16量子化を使用しますか (y/N, デフォルト: {'Y' if USE_FP16 else 'N'}): ").lower()
    fp16_to_use = USE_FP16
    if use_fp16_input == 'y':
        fp16_to_use = True
    elif use_fp16_input == 'n':
        fp16_to_use = False
    
    # INT8は通常、追加のキャリブレーションデータセットが必要なため、デフォルトでは無効
    # 必要に応じて有効化してください。
    # use_int8_input = input(f"INT8量子化を使用しますか (y/N, デフォルト: {'Y' if USE_INT8 else 'N'}): ").lower()
    # int8_to_use = USE_INT8
    # if use_int8_input == 'y':
    #     int8_to_use = True
    # elif use_int8_input == 'n':
    #     int8_to_use = False
    int8_to_use = USE_INT8 # INT8は現時点ではデフォルトFalseのまま

    convert_model_to_tensorrt(
        model_path=model_to_convert,
        output_name=OUTPUT_ENGINE_NAME,
        imgsz=imgsz_to_use,
        half=fp16_to_use,
        int8=int8_to_use,
        simplify=SIMPLIFY_ONNX
    )
