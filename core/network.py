import os
import warnings
from dataclasses import dataclass

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from config.config import DATES
from utils.create_gif import create_gif
from utils.delete_files import delete_files

raw_links_dir = os.path.join(os.getcwd(), "data", "01_raw", "links")
output_network_dir = os.path.join(
    os.getcwd(), "data", "02_output", "network_incidences"
)
output_plots_dir = os.path.join(os.getcwd(), "data", "02_output", "network_plots")


@dataclass
class Network:
    """国・県が保有する道路網遮断情報とリンクデータを組み合わせてネットワークを生成（csv）、可視化（tiff）"""

    # ネットワーク生成の指定日
    dates: list[str]

    # ネットワークのリンク（LineString）を表現するDataframeのカラム
    link_coord_cols = ["longitude1", "longitude2", "latitude1", "latitude2"]

    # リンクデータ（shp）の読み込み
    gdf_link: gpd.GeoDataFrame = gpd.read_file(
        os.path.join(raw_links_dir, "link_data.shp")
    )

    # 県（pref）の道路網遮断情報データを読み込み
    df_disrp_pref: pd.DataFrame = pd.read_csv(
        os.path.join(raw_links_dir, "disrupted_link_prefectural.csv")
    )

    # 国（nat）の道路網遮断情報データを読み込み
    df_disrp_nat: pd.DataFrame = pd.read_csv(
        os.path.join(raw_links_dir, "disrupted_link_national.csv")
    )

    def __post_init__(self) -> None:
        """データ前処理（インスタンス初期化後に実行される）"""
        # # ディレクトリにあるファイルを削除
        delete_files(directory=output_network_dir, extensions=["csv"])
        delete_files(directory=output_plots_dir, extensions=["tiff", "gif"])

        # 道路網遮断データと結合するためにリンクのカラムを少数4桁で丸める
        self.gdf_link[self.link_coord_cols] = self.gdf_link[self.link_coord_cols].round(
            4
        )

        # ノードの最長文字数
        self.max_node_char_len = max(
            self.gdf_link[["node1", "node2"]].astype(str).map(len).max()
        )

        # linkカラムの作成
        self.gdf_link["link"] = (
            self.gdf_link["node1"] * 10**self.max_node_char_len + self.gdf_link["node2"]
        ).astype(str)

        # リンクデータ（gdf_link）と結合するためにリンクのカラムを少数4桁で丸める
        self.df_disrp_nat[self.link_coord_cols] = self.df_disrp_nat[
            self.link_coord_cols
        ].round(4)

    def merge_disrp_nat_into_link(self) -> pd.DataFrame:
        """リンクに国の道路網遮断情報（LineString）をマージ

        Returns:
            pd.DataFrame: マージ後のリンクDataframe
        """
        df_network = pd.merge(
            self.gdf_link,
            self.df_disrp_nat[["start_time", "end_time"] + self.link_coord_cols],
            on=self.link_coord_cols,
            how="left",
        )

        df_network["key"] = (
            df_network[self.link_coord_cols].astype(str).agg(" ".join, axis=1)
        )

        return df_network

    def merge_disrp_pref_into_network(self, df_network: pd.DataFrame) -> pd.DataFrame:
        """ネットワークに県の道路網遮断情報（Point）をマージ

        Returns:
            pd.DataFrame: マージ後のDataframe
        """
        # df_disrp_prefをGeoDataFrameに変換
        gdf_disrp_pref_points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                self.df_disrp_pref["longitude"], self.df_disrp_pref["latitude"]
            ),
            crs=self.gdf_link.crs,
        )

        warnings.filterwarnings("ignore", category=UserWarning)

        # 道路網遮断ポイントと最も近いリンクを抽出
        gdf_link_nearest = gdf_disrp_pref_points.apply(
            lambda row: self.gdf_link.loc[
                self.gdf_link.geometry.distance(row.geometry).idxmin()
            ],
            axis=1,
        )

        warnings.filterwarnings("default", category=UserWarning)

        gdf_link_nearest = pd.concat(
            [
                self.df_disrp_pref,
                gdf_link_nearest,
            ],
            axis=1,
        )

        disrp_pref_cols = [
            "ID",
            "start_time",
            "end_time",
            "reason",
            "weather",
            "regulation",
        ]

        # 最も近いLineが２方向か１方向かを確認
        df_combined_directions = pd.concat(
            [
                gdf_link_nearest[["node1", "node2"] + disrp_pref_cols],
                gdf_link_nearest[["node2", "node1"] + disrp_pref_cols].rename(
                    columns={"node2": "node1", "node1": "node2"}
                ),
            ],
            ignore_index=True,
        ).merge(self.gdf_link, on=["node1", "node2"])

        df_disrp_pref_link = df_combined_directions[
            disrp_pref_cols + self.link_coord_cols
        ]

        df_disrp_pref_link = df_disrp_pref_link.copy()
        df_disrp_pref_link["key"] = (
            df_disrp_pref_link[self.link_coord_cols].astype(str).agg(" ".join, axis=1)
        )

        df_disrp_pref_link = df_disrp_pref_link.rename(
            {"start_time": "start_pref", "end_time": "end_pref"}, axis="columns"
        )

        df_network = pd.merge(
            df_network,
            df_disrp_pref_link[["key", "start_pref", "end_pref"]],
            on="key",
            how="left",
        )

        df_network["start_time"] = df_network.apply(
            lambda row: (
                row["start_time"] if pd.notna(row["start_time"]) else row["start_pref"]
            ),
            axis=1,
        )
        df_network["end_time"] = df_network.apply(
            lambda row: (
                row["end_time"] if pd.notna(row["end_time"]) else row["end_pref"]
            ),
            axis=1,
        )

        # 不要なカラムをドロップ
        df_network = df_network.drop(columns=["key", "start_pref", "end_pref"])

        return df_network

    def save_incidence_matrix(
        self, df: pd.DataFrame, date: pd.Timestamp, output_dir: str
    ) -> None:
        """dateの日にちについての接続行列（incidence matrix）を作成し、csv保存

        Args:
            df (pd.DataFrame): ネットワークのDataframe
            date (pd.Timestamp): date
            output_dir (str): 出力先ディレクトリ
        """
        df_a = df.copy()
        df_b = df.copy()

        df_link_incidence = pd.merge(
            df_a, df_b, left_on="node2", right_on="node1", how="outer"
        )

        # カラム名に '_x' がつくものを '_o' にリネーム
        df_link_incidence = df_link_incidence.rename(
            columns=lambda col: col.replace("_x", "_o") if "_x" in col else col,
        )
        # カラム名に '_y' がつくものを '_d' にリネーム
        df_link_incidence = df_link_incidence.rename(
            columns=lambda col: col.replace("_y", "_d") if "_y" in col else col,
        )

        df_link_incidence["date"] = pd.to_datetime(date).date()
        df_link_incidence["flag"] = (
            df_link_incidence["flag_o"] | df_link_incidence["flag_d"]
        ).astype(int)
        df_link_incidence["flag"] = df_link_incidence["flag"].fillna(0).astype(int)

        # 必要なカラムのキーワード
        col_keywords = [
            "node",
            "link",
            "lklength",
            "start_time",
            "end_time",
            "mrdclasscd",
            "speed_",
            "direction",
            "longitude",
            "latitude",
            "flag",
        ]

        filtered_cols = [
            col
            for col in df_link_incidence.columns
            if any(keyword in col for keyword in col_keywords) or col == "date"
        ]

        df_link_incidence = df_link_incidence[filtered_cols]

        # ネットワークの接続行列（incidence matrix）をCSVに保存
        df_link_incidence.to_csv(
            os.path.join(output_dir, f"network_{date}.csv"),
            index=False,
        )

    def plot_network(
        self, df: pd.DataFrame, date: pd.Timestamp, output_dir: str
    ) -> None:
        """ある日のネットワークをプロットし、TIFFを保存

        Args:
            df (pd.DataFrame): longitude1, latitude1, longitude2, latitude2カラムを持つDataframe
            date (pd.Timestamp): 日
            output_dir (str): 出力先ディレクトリ
        """
        # PointとLineStringを作成
        points1 = [Point(xy) for xy in zip(df["longitude1"], df["latitude1"])]
        points2 = [Point(xy) for xy in zip(df["longitude2"], df["latitude2"])]
        lines = [
            LineString(xy)
            for xy in zip(
                zip(df["longitude1"], df["latitude1"]),
                zip(df["longitude2"], df["latitude2"]),
            )
        ]

        # ノードとリンクのGeoDataframeを作成
        gdf_points1 = gpd.GeoDataFrame(geometry=points1)
        gdf_points2 = gpd.GeoDataFrame(geometry=points2)
        gdf_lines = gpd.GeoDataFrame(geometry=lines)

        # プロット
        _, ax = plt.subplots(figsize=(10, 8))
        pd.concat([gdf_points1, gdf_points2], axis=0).plot(
            ax=ax, color="blue", markersize=30, label="Node"
        )
        gdf_lines.plot(ax=ax, color="gray", linewidth=2, alpha=0.5, label="Link")
        ax.set_title(f"Network connectivity {date}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        plt.savefig(os.path.join(output_dir, f"network_{date}.tiff"), dpi=300)
        plt.close()

    def process(self) -> None:
        """ネットワークを生成"""
        # リンクデータに国・県の道路網切断情報をマージしてネットワークデータを生成
        df_network = self.merge_disrp_nat_into_link()
        df_network = self.merge_disrp_pref_into_network(df_network=df_network)

        df_network["date_start"] = pd.to_datetime(df_network["start_time"]).dt.date
        df_network["date_end"] = pd.to_datetime(df_network["end_time"]).dt.date

        for date in self.dates:
            date = pd.to_datetime(date).date()

            df_network_date = df_network.copy()
            df_network_date["date"] = pd.to_datetime(date).date()

            # flag: dateがdate_startとdate_endの間にあるかどうか（道路網が遮断されているかどうか）
            df_network_date["flag"] = np.where(
                (
                    df_network_date["date_start"] - df_network_date["date"]
                    <= pd.Timedelta(days=0)
                )
                & (
                    df_network_date["date_end"] - df_network_date["date"]
                    >= pd.Timedelta(days=0)
                ),
                1,
                0,
            )

            # ネットワークの接続行列（incidence matrix）をcsv保存
            self.save_incidence_matrix(
                df=df_network_date, date=date, output_dir=output_network_dir
            )

            # dateの日に遮断されていないネットワークのみを抽出
            df_network_not_disrupted = df_network_date[df_network_date["flag"] == 0]
            # ネットワークのTIFFを保存
            self.plot_network(
                df=df_network_not_disrupted, date=date, output_dir=output_plots_dir
            )

        # 日毎のTIFFファイルからGIFを生成
        tiff_files = sorted(
            [f for f in os.listdir(output_plots_dir) if f.endswith(".tiff")]
        )
        create_gif(
            output_dir=output_plots_dir,
            files=tiff_files,
            gif_name="networks.gif",
        )


if __name__ == "__main__":
    network = Network(dates=DATES)
    network.process()
