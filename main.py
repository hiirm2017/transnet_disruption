import os
import time

from loguru import logger
from natsort import natsorted

from config.config import DATES, MIN_SPEED, RL_PARAMS
from core.logsum_calc import LogsumCalc
from core.logsum_viz import LogsumViz
from core.network import Network

output_network_dir = os.path.join(
    os.getcwd(), "data", "02_output", "network_incidences"
)
output_logsum_dir = os.path.join(os.getcwd(), "data", "02_output", "logsums")
output_link_dir = os.path.join(os.getcwd(), "data", "02_output", "links")


def main() -> None:
    """全ての処理を実行"""

    logger.info("Network process started")
    network = Network(dates=DATES)
    network.process()
    logger.info("Network process done")

    # data/02_output/network_incidenceへのcsv出力完了を待つ
    while True:
        files = os.listdir(output_network_dir)
        csv_files = [file for file in files if file.endswith(".csv")]
        if len(csv_files) == len(DATES):
            break
        time.sleep(5)

    time.sleep(10)

    network_files = natsorted(
        [f for f in os.listdir(output_network_dir) if f.endswith(".csv")]
    )

    logger.info("LogsumCalc process started")
    logsum_calc = LogsumCalc(
        min_speed=MIN_SPEED, rl_params=RL_PARAMS, network_files=network_files
    )
    logsum_calc.process()
    logger.info("LogsumCalc process done")

    # data/02_output/<logsums | links>へのcsv出力完了を待つ
    while True:
        files = os.listdir(output_logsum_dir)
        files2 = os.listdir(output_link_dir)
        csv_files = [file for file in files if file.endswith(".csv")]
        csv_files2 = [file for file in files2 if file.endswith(".csv")]
        if len(csv_files) == len(DATES) and len(csv_files2) == len(DATES):
            break
        time.sleep(5)

    logsum_files = natsorted(
        [f for f in os.listdir(output_logsum_dir) if f.endswith(".csv")]
    )
    link_files = natsorted(
        [f for f in os.listdir(output_link_dir) if f.endswith(".csv")]
    )

    logger.info("LogsumViz process started")
    logsum_viz = LogsumViz(logsum_files=logsum_files, link_files=link_files)
    logsum_viz.process()
    logger.info("LogsumViz process done")


if __name__ == "__main__":
    main()
