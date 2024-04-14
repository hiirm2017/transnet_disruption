# 道路網切断に伴う道路網性能低下の定量評価

## ソースコード概要

本ソースコードは以下を実施しています。

- 道路網切断情報をDRMデータと統合（`core/network.py`）
- 道路網切断に伴う道路網性能低下を定量評価（`core/logsum_calc.py`、`core/logsum_viz.py`）

道路網性能低下の定量評価は，Recursive logitモデルから得られるログサム値を用いています。詳しくは，Safitri and Chikaraishi (2022)をご確認ください。

### 道路網切断情報をDRMデータと統合

- 道路網の切断は，リンクIDで管理された切断情報（`data/01_raw/links/disrupted_link_national.csv`）と，緯度軽度で管理された切断情報（`data/01_raw/links/disrupted_link_prefectural.csv`）の２種類を想定しています。

![disruption_animation](/docs/networks.gif)

### 道路網切断に伴う道路網性能低下を定量評価

![logsum_animation](/docs/logsums.gif)

## ディレクトリ構成

```bash
.
├── Dockerfile
├── README.md
├── config
│   └── config.py
├── core
│   ├── __init__.py
│   ├── logsum_calc.py
│   ├── logsum_viz.py
│   └── network.py
├── data
│   ├── 01_raw
│   │   ├── links
│   │   │   ├── disrupted_link_national.csv
│   │   │   ├── disrupted_link_prefectural.csv
│   │   │   ├── link_data.cpg
│   │   │   ├── link_data.dbf
│   │   │   ├── link_data.prj
│   │   │   ├── link_data.sbn
│   │   │   ├── link_data.sbx
│   │   │   ├── link_data.shp
│   │   │   └── link_data.shx
│   │   └── nodes
│   │       ├── node_data.dbf
│   │       ├── node_data.prj
│   │       ├── node_data.sbn
│   │       ├── node_data.sbx
│   │       ├── node_data.shp
│   │       ├── node_data.shx
│   │       └── od.csv
│   └── 02_output
│       ├── links
│       ├── logsum_plots
│       ├── logsums
│       ├── network_incidences
│       └── network_plots
├── docs
│   ├── logsums.gif
│   └── networks.gif
├── main.py
├── poetry.lock
├── pyproject.toml
└── utils
    ├── __init__.py
    ├── create_gif.py
    └── delete_files.py
```

## プログラム実行方法

### Docker

Dockerを起動し、以下のコマンドを実行することで、Dockerコンテナを作成してコンテナの中に入ります。

```bash
$ docker build -t network-disrp .
```

```bash
# for Mac , Linux
$ docker run -it --rm -v "$(pwd):/app" --entrypoint /bin/bash network-disrp

# for Windows
$ docker run -it --rm -v "%cd%:/app" --entrypoint /bin/bash network-disrp
```

コンテナの中に入ったのち、`poetry run python main.py`コマンドでプログラムを実行します。

```bash
root@8e54d036694b:/app# poetry run python main.py
```

コンテナを出る時は`exit`を実行します。

```bash
root@8e54d036694b:/app# exit
```

### ローカル（動作未保証）

以下のコマンドで`main.py`を実行します。

```bash
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install --upgrade pip
$ pip install poetry
$ poetry install --no-root
$ python main.py
```

## お問い合わせ

- 力石真 (Makoto Chikaraishi)
  - chikaraishim@hiroshima-u.ac.jp
- Diana Nur Safitri
  - dianas@hiroshima-u.ac.jp

本研究は，日本デジタル道路地図協会令和５年度研究助成「道路ネットワーク性能ダイナミクスの指標化及び可視化に関する研究」のもと実施したものです。

参考文献

- [Safitri, N.D., Chikaraishi, M. (2022) Impact of Transport Network Disruption on Travel Demand: A Case Study of July 2018 Heavy Rain Disaster, Japan, Asian Transport Studies, 8, 100057.](https://www.sciencedirect.com/science/article/pii/S2185556022000037)
