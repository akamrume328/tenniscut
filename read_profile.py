import pstats
import sys

def read_profile_stats(profile_file_path):
    """
    .prof ファイルを読み込み、内容をテキストとして表示するスクリプト
    """
    # 引数で渡されたファイル名を表示
    print("\n" + "="*50)
    print(f"--- プロファイル結果: {profile_file_path} ---")
    print("--- (cumtime: 関数が消費した合計時間でソート) ---")
    print("="*50 + "\n")
    
    try:
        # プロファイルファイルを読み込む
        stats = pstats.Stats(profile_file_path)
        
        # 不要なパス情報を削除して見やすくし、処理時間が長い順にソートして、上位75件を表示
        stats.strip_dirs().sort_stats('cumulative').print_stats(75)

    except FileNotFoundError:
        print(f"エラー: 指定されたファイルが見つかりません: '{profile_file_path}'")
        print("ファイルパスが正しいか、ファイルがこのディレクトリに存在するか確認してください。")
    except Exception as e:
        print(f"プロファイル読み込み中にエラーが発生しました: {e}")

if __name__ == '__main__':
    # コマンドラインからファイル名を受け取る
    if len(sys.argv) < 2:
        print("エラー: プロファイルファイル名を指定してください。")
        print("使い方: python read_profile.py <.prof ファイル名>")
        sys.exit(1)
        
    target_file = sys.argv[1]
    read_profile_stats(target_file)