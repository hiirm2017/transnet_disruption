import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from natsort import natsorted

from utils.create_gif import create_gif
from utils.delete_files import delete_files

output_logsum_dir = os.path.join(os.getcwd(), "data", "02_output", "logsums")
output_link_dir = os.path.join(os.getcwd(), "data", "02_output", "links")
output_logsum_plot_dir = os.path.join(os.getcwd(), "data", "02_output", "logsum_plots")


@dataclass
class LogsumViz:
    """ログサムを可視化"""

    # ソートしたログサム（csv）のファイル
    logsum_files: list[str]

    # ソートしたリンク（csv）のファイル
    link_files: list[str]

    def __post_init__(self) -> None:
        """前処理（インスタンス初期化後に実行）"""
        # ディレクトリにあるファイルを削除
        delete_files(directory=output_logsum_plot_dir, extensions=["png", "gif"])

        self.dfs_logsum: list[pd.DataFrame] = []

        for i, lf in enumerate(self.logsum_files):
            df_logsum = pd.read_csv(os.path.join(output_logsum_dir, lf))
            df_link = pd.read_csv(os.path.join(output_link_dir, self.link_files[i]))

            # ログサムとリンクリストを横に結合
            df = pd.concat([df_link, df_logsum], axis=1)

            df = df.dropna()
            df["link"] = df["link"].astype(int).astype(str)
            df["origin_area"] = df["link"].apply(lambda x: int(x[-2:]))

            df = df.sort_values(by="origin_area")
            self.dfs_logsum.append(df)
            self.dfs_logsum[i]["source"] = lf

    def process(self) -> None:
        """ログサムを可視化"""

        df_logsums = pd.concat(self.dfs_logsum, ignore_index=True)

        logsum_cols = sorted([col for col in df_logsums if "V" in col])

        df_logsums = df_logsums[logsum_cols]

        # Legendの最大値
        max_value = df_logsums.replace(np.inf, np.nan).max().max()

        # Legendの最小値
        min_value = df_logsums.replace(-np.inf, np.nan).min().min()

        for df in self.dfs_logsum:
            date = re.search(r"\d{4}-\d{2}-\d{2}", df["source"].iloc[0]).group()  # type: ignore

            df = df.set_index("origin_area")
            df = df[logsum_cols]

            df.columns = df.index.tolist()

            df_stack = df.stack().reset_index()
            df_stack.columns = ["row_names", "col_names", "value"]

            df_stack["col_names"] = df_stack["col_names"].astype(int)
            df_stack["value"] = np.where(
                df_stack["row_names"] == df_stack["col_names"], 0, df_stack["value"]
            )

            df_pivot = df_stack.pivot_table(
                index="row_names", columns="col_names", values="value"
            )

            # 値が0のマスをマスクするためにマスクする行列を抽出
            df_mask = df_stack.pivot(
                index="row_names", columns="col_names", values="value"
            ).eq(0)

            # プロット
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                df_pivot,
                cmap="Reds",
                vmin=min_value,
                vmax=max_value,
                center=0,
                mask=df_mask,
                annot=True,
                cbar_kws={"ticks": np.arange(min_value, max_value + 1, 10)},
            )
            plt.title(f"Logsum value {date}")
            plt.xlabel("Destination")
            plt.ylabel("Origin")
            plt.gca().invert_yaxis()  # y軸で逆転
            plt.savefig(
                os.path.join(output_logsum_plot_dir, f"logsum_{date}.png"),
                format="png",
                dpi=300,
            )
            plt.close()

        # GIF生成
        png_files = sorted(
            [
                file
                for file in os.listdir(output_logsum_plot_dir)
                if file.endswith(".png")
            ]
        )
        create_gif(
            output_dir=output_logsum_plot_dir, files=png_files, gif_name="logsums.gif"
        )


if __name__ == "__main__":
    logsum_files = sorted(
        [f for f in os.listdir(output_logsum_dir) if f.endswith(".csv")]
    )

    link_files = natsorted(
        [f for f in os.listdir(output_link_dir) if f.endswith(".csv")]
    )

    logsum_viz = LogsumViz(logsum_files=logsum_files, link_files=link_files)
    logsum_viz.process()
