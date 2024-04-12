import os
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.sparse import csr_matrix, identity, lil_matrix

from config.config import MIN_SPEED, RL_PARAMS
from utils.delete_files import delete_files

raw_nodes_dir = os.path.join(os.getcwd(), "data", "01_raw", "nodes")
output_network_dir = os.path.join(
    os.getcwd(), "data", "02_output", "network_incidences"
)
output_link_dir = os.path.join(os.getcwd(), "data", "02_output", "links")
output_logsum_dir = os.path.join(os.getcwd(), "data", "02_output", "logsums")


@dataclass
class LogsumCalc:
    """ログサム計算"""

    # 最小スピード
    min_speed: int
    # recursive logitのパラメータ
    rl_params: list[int]

    # zoneデータの読み込み
    df_zone = pd.read_csv(os.path.join(raw_nodes_dir, "od.csv"))

    # nodeデータの読み込み
    gdf_node = gpd.read_file(os.path.join(raw_nodes_dir, "node_data.shp"))

    # ネットワークのファイル
    network_files: list[str]

    def __post_init__(self) -> None:
        """前処理（インスタンス初期化後に実行）"""
        # ディレクトリにあるファイルを削除
        delete_files(directory=output_logsum_dir, extensions=["csv"])
        delete_files(directory=output_link_dir, extensions=["csv"])

        # geometryカラムからlongitudeとlatitudeを取得
        self.gdf_node["longitude"] = self.gdf_node.geometry.x
        self.gdf_node["latitude"] = self.gdf_node.geometry.y

        # 必要なカラムのみ抽出
        self.gdf_node = self.gdf_node[["node", "longitude", "latitude"]]

    def euclidean_distance(
        self, lon1: np.float_, lat1: np.float_, lon2: np.float_, lat2: np.float_
    ) -> np.float_:
        """ユークリッド距離計算

        Args:
            lon1 (np.float_): longitude 1
            lat1 (np.float_): latitude 1
            lon2 (np.float_): longitude 2
            lat2 (np.float_): latitude 2

        Returns:
            np.float_: ユークリッド距離
        """

        return np.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)

    # Function to find the nearest node
    def find_nearest_nodes(
        self, df_zone: pd.DataFrame, gdf_node: gpd.GeoDataFrame
    ) -> np.ndarray[int]:
        """df_zoneについてgdf_nodeから最も近いノードを取得

        Args:
            df_zone (pd.DataFrame): df_zone
            gdf_node (gpd.GeoDataFrame): gdf_node

        Returns:
            np.ndarray[int]: 最も近いノードのリスト（例：[9 3 13 11 5 1])
        """
        nearest_node = np.empty(len(df_zone), dtype=int)

        for j in range(len(df_zone)):
            # 最短距離とnearestノードを格納するために変数を初期化
            min_distance: Any = np.inf

            zone_lon = df_zone["longitude"].iloc[j]
            zone_lat = df_zone["latitude"].iloc[j]

            for k in range(len(gdf_node)):
                node_lon = gdf_node["longitude"].iloc[k]
                node_lat = gdf_node["latitude"].iloc[k]

                # ユークリッド距離
                distance = self.euclidean_distance(
                    lon1=zone_lon, lat1=zone_lat, lon2=node_lon, lat2=node_lat
                )

                # 最短距離とnearestノードを更新
                if distance < min_distance:
                    min_distance = distance
                    nearest_node[j] = gdf_node["node"].iloc[k]

        return nearest_node

    def add_variables_to_network(self, df: pd.DataFrame) -> pd.DataFrame:
        """ネットワークのDataframeに変数（カラム）を追加

        Args:
            df (pd.DataFrame): ネットワークのDataframe

        Returns:
            pd.DataFrame: Dataframe
        """
        df["lklength_d"] = np.where(df["lklength_d"].isna(), 0, df["lklength_d"])

        df["speed_d"] = np.where(
            (df["speed_d"] == 0) | df["speed_d"].isna(), self.min_speed, df["speed_d"]
        )

        df["TT_min"] = 60 * df["lklength_d"] * (1 / 1000) / df["speed_d"]

        df["mrdclasscd_o"] = df["mrdclasscd_o"].apply(
            lambda x: 9 if x == 0 or pd.isna(x) else x
        )
        df["mrdclasscd_d"] = df["mrdclasscd_d"].apply(
            lambda x: 9 if x == 0 or pd.isna(x) else x
        )

        df["cost_yen"] = 0
        df["cost_yen"] = np.where(
            (df["mrdclasscd_o"] > 2) & (df["mrdclasscd_d"] <= 2),
            (150 + (df["lklength_d"] / 1000) * 24.6) * 1.1,
            np.where(
                df["mrdclasscd_d"] <= 2,
                ((df["lklength_d"] / 1000) * 24.6) * 1.1,
                0,
            ),
        )
        df["cost_yen"] += df["lklength_d"] * (1 / 1000) * (150 / 5)
        df["cost_yen"] = df["cost_yen"].fillna(0).astype(float)

        df["uturndummy"] = np.where(df["node1_o"] == df["node2_d"], 1, 0)
        df["uturndummy"] = df["uturndummy"].fillna(0).astype(int)

        return df

    def compute_logsum(self, df: pd.DataFrame) -> np.ndarray[float]:
        """ログサムを計算

        Args:
            df (pd.DataFrame): ネットワークのDataframe

        Returns:
            np.ndarray[float]: ログサム計算結果
        """
        dest_links = df[df["destination_flag"] == 1]["link_d"].unique()
        max_link = max(df["link_o"].max(), df["link_d"].max())

        IM = identity(max_link).toarray()

        B = lil_matrix((max_link, len(dest_links)), dtype=int).toarray()
        for j, i in enumerate(dest_links):
            B[i - 1, j] = 1

        M = np.exp(
            self.rl_params[0] * df["TT_min"]
            + self.rl_params[1] * df["cost_yen"] / 10
            + self.rl_params[2] * df["uturndummy"]
        )

        SM = csr_matrix(
            (M, ((df["link_o"] - 1), (df["link_d"] - 1))),
            shape=(max_link, max_link),
        ).toarray()

        # 指定された行を0に設定
        SM[(dest_links - 1), :] = 0

        z = np.dot(np.linalg.inv(IM - SM), B)

        np.seterr(divide="ignore", invalid="ignore")

        logsum_value = np.log(z)

        return logsum_value

    def process(self) -> None:
        """ログサムを計算"""

        dfs_network = [
            pd.read_csv(os.path.join(output_network_dir, network_file))
            for network_file in self.network_files
        ]

        # df_zoneについてgdf_nodeから最も近いノードをnearest_nodeカラムの格納
        self.df_zone["nearest_node"] = self.find_nearest_nodes(
            df_zone=self.df_zone, gdf_node=self.gdf_node
        )

        logsums = []

        for i, l in enumerate(dfs_network):
            df_network = pd.DataFrame(l)

            # originのダミーリンク
            df_dummy_o = pd.merge(
                df_network,
                self.df_zone[["nearest_node", "area_name"]],
                left_on="node1_o",
                right_on="nearest_node",
            ).drop(columns=["nearest_node"])

            df_dummy_o = df_dummy_o.drop_duplicates(
                subset="link_o", keep="first"
            ).reset_index(drop=True)

            max_node_num = max(
                df_dummy_o[["node1_o", "node1_d", "node2_d"]].values.flatten()
            )
            max_node_char_num = max([len(str(max_node_num))])

            for col in df_dummy_o.filter(like="_o").columns:
                col_d = col.replace("_o", "_d")
                df_dummy_o[col_d] = df_dummy_o[col]
                df_dummy_o[col] = np.nan

            o_link_cols = ["link_o", "node1_o", "node2_o"]
            df_dummy_o[o_link_cols] = df_dummy_o[o_link_cols].astype(str)
            for node in df_dummy_o["node1_d"].unique():
                max_node_num += 1
                df_dummy_o.loc[df_dummy_o["node1_d"] == node, "link_o"] = str(
                    max_node_num
                ) + node.astype(str).zfill(max_node_char_num)
                df_dummy_o.loc[df_dummy_o["node1_d"] == node, "node1_o"] = str(
                    max_node_num
                )
                df_dummy_o.loc[df_dummy_o["node1_d"] == node, "node2_o"] = str(node)

            df_dummy_o["flag"] = df_dummy_o["flag_d"]

            # destinationのダミーリンク
            df_dummy_d = pd.merge(
                df_network,
                self.df_zone[["nearest_node", "area_name"]],
                left_on="node2_d",
                right_on="nearest_node",
            )

            df_dummy_d = df_dummy_d.drop(columns=["nearest_node"])

            df_dummy_d = df_dummy_d.drop_duplicates(
                subset="link_d", keep="first"
            ).reset_index(drop=True)

            for col in df_dummy_d.filter(like="_d").columns:
                col_o = col.replace("_d", "_o")
                df_dummy_d[col_o] = df_dummy_d[col]
                df_dummy_d[col] = np.nan

            d_link_cols = ["link_d", "node1_d", "node2_d"]
            df_dummy_d[d_link_cols] = df_dummy_d[d_link_cols].astype(str)
            for node in df_dummy_d["node2_o"].unique():
                max_node_num += 1
                df_dummy_d.loc[df_dummy_d["node2_o"] == node, "link_d"] = (
                    "1" + node.astype(str).zfill(max_node_char_num) + str(max_node_num)
                )
                df_dummy_d.loc[df_dummy_d["node2_o"] == node, "node2_d"] = str(
                    max_node_num
                )
                df_dummy_d.loc[df_dummy_d["node2_o"] == node, "node1_d"] = str(node)

            df_dummy_d["flag"] = df_dummy_d["flag_o"]

            df_network["area_name"] = 0

            network_spec_dat = pd.concat(
                [df_network, df_dummy_o, df_dummy_d], ignore_index=True
            )
            network_spec_dat["origin_flag"] = np.where(
                network_spec_dat["lklength_o"].isna(), 1, 0
            )
            network_spec_dat["destination_flag"] = np.where(
                network_spec_dat["lklength_d"].isna(), 1, 0
            )

            # 必要な変数を追加
            network_spec_dat = self.add_variables_to_network(df=network_spec_dat)

            network_spec_dat = network_spec_dat[network_spec_dat["flag"] == 0]
            network_spec_dat["link_o"] = network_spec_dat["link_o"].astype(int)
            network_spec_dat["link_d"] = network_spec_dat["link_d"].astype(int)

            # リンクを再ナンバリング
            df_link = pd.DataFrame(
                data={
                    "link": pd.concat(
                        [network_spec_dat["link_o"], network_spec_dat["link_d"]]
                    )
                    .drop_duplicates()
                    .sort_values()
                    .astype(str)
                    .reset_index(drop=True)
                }
            )
            df_link["new_link"] = range(1, len(df_link) + 1)

            origin = pd.DataFrame(data={"link": df_dummy_o["link_o"].unique()})

            origin["area"] = df_dummy_o["area_name"].unique()

            df_link["link"] = df_link["link"].astype(str)
            df_link = pd.merge(
                df_link, origin, left_on="link", right_on="link", how="left"
            )[["link", "new_link", "area"]]

            df_link.to_csv(
                os.path.join(output_link_dir, f"old_new_link_{i + 1}.csv"),
                index=False,
            )

            network_spec_dat["link_o"] = network_spec_dat["link_o"].astype(str)

            network = pd.merge(
                network_spec_dat,
                df_link,
                left_on="link_o",
                right_on="link",
            )

            network["link_o"] = network["link_o"].astype(int)
            network = network.drop(columns=["link"])
            network["link_o"] = network["new_link"]
            network = network.drop(columns=["new_link"])
            network["link_d"] = network["link_d"].astype(str)

            network = pd.merge(network, df_link, left_on="link_d", right_on="link")

            network["link_d"] = network["link_d"].astype(int)
            network = network.sort_values(by=["link_d", "link"]).drop(columns=["link"])
            network["link_d"] = network["new_link"]
            network = network.drop(columns=["new_link"])

            logsum = self.compute_logsum(df=network)

            logsums.append(logsum)

        for i, df in enumerate(dfs_network):
            date = pd.DataFrame(df)["date"].iloc[0]

            col_length = logsums[i].shape[1]

            df_logsum_dense = pd.DataFrame(
                logsums[i], columns=[f"V{col_id+1}" for col_id in range(col_length)]
            )

            df_logsum_dense.to_csv(
                os.path.join(output_logsum_dir, f"logsum_{date}.csv"), index=False
            )


if __name__ == "__main__":
    network_files = natsorted(
        [f for f in os.listdir(output_network_dir) if f.endswith(".csv")]
    )
    logsum_calc = LogsumCalc(
        min_speed=MIN_SPEED, rl_params=RL_PARAMS, network_files=network_files
    )
    logsum_calc.process()
