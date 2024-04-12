import os

from PIL import Image


def create_gif(output_dir: str, files: list[str], gif_name: str) -> None:
    """GIFを生成する

    Args:
        output_dir (str): 出力先directory
        files (list[str]): ファイル名一覧 (png, tiffなど、PIL.Imageで読み込めるデータのみ対応)
        gif_name (str): 出力GIF名
    """
    images = [Image.open(os.path.join(output_dir, f)) for f in files]
    images[0].save(
        os.path.join(output_dir, gif_name),
        save_all=True,
        append_images=images[1:],
        duration=1000,
        loop=0,
    )
