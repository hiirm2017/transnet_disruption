import glob
import os


def delete_files(directory: str, extensions: list[str]) -> None:
    """_summary_

    Args:
        directory (str): _description_
        extension (list[str]): _description_
    """
    # ディレクトリ内の指定された拡張子のファイル一覧を取得
    files_to_delete = []
    for extension in extensions:
        files_to_delete.extend(glob.glob(os.path.join(directory, f"*.{extension}")))

    # ファイルを削除
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"エラー: {file_path} を削除できませんでした。{e}")
