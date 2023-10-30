import requests
import pandas as pd
import sys
import time
from units import getConfig
import warnings
from tqdm import tqdm

pd.set_option("mode.chained_assignment", None)
warnings.filterwarnings("ignore")
from loguru import logger

def main():
    # 配置信息
    video_record_add = getConfig("gnn_spider", "record_address")
    total_videoinfo_add = getConfig("gnn_spider", "videoinfo_address")
    data=pd.read_excel(total_videoinfo_add,dtype=str)
    data=data.drop_duplicates(subset='aweme_id')
    rawdata=pd.read_excel(video_record_add,dtype=str)
    rawdata=rawdata[~rawdata['aweme_id'].isin(data.aweme_id)]
    rawdata = rawdata.drop_duplicates(subset="aweme_id")[::-1]
    token = getConfig("gnn_spider", "token")
    total_videoinfo_num = 0
    try:
        total_videoinfo = pd.read_excel(total_videoinfo_add, dtype=str)
        total_videoinfo_num = len(total_videoinfo)
    except:
        total_videoinfo = pd.DataFrame(
            columns=[]
        )

    payload = {}
    headers = {"User-Agent": "Apifox/1.0.0 (https://www.apifox.cn)"}
    failed_target = 0
    trace_list = []
    trace_list.append(
        logger.add(
            "log/VideoinfoInfo.log",
            filter=lambda record: record["extra"]["name"] == "VideoinfoInfo",
        )
    )
    logger_info = logger.bind(name="VideoinfoInfo")

    ## 爬取页号
    total_videoinfo_list = pd.DataFrame()
    for index, row in tqdm(rawdata.iterrows()):
        url = (
            f"http://ttt.258data.com/dy/video/info?video_id={row.aweme_id}&token={token}"
        )
        response = requests.request("GET", url, headers=headers, data=payload)
        scrapy_videoinfo_list = response.json()
        if scrapy_videoinfo_list["code"] == 0:
            failed_target = 0
            if scrapy_videoinfo_list["aweme_detail"] != None:
                repeat_inspect = pd.DataFrame([scrapy_videoinfo_list["aweme_detail"]])
                total_videoinfo_list = total_videoinfo_list.append(repeat_inspect)
                logger_info.info(f"Get videoinfo {index} Success")
                time.sleep(2)
        else:  # 请求失败 返回页号
            time.sleep(10)
            continue
            for i in range(3):
                response = requests.request("GET", url, headers=headers, data=payload)
                scrapy_videoinfo_list = response.json()
                if scrapy_videoinfo_list["code"] == 0:
                    failed_target = 0
                    logger_info.info("Failed Retry, Success")  ## 请求返回成功了
                    if scrapy_videoinfo_list["aweme_detail"] != None:
                        repeat_inspect = pd.DataFrame([scrapy_videoinfo_list["aweme_detail"]])
                        total_videoinfo_list = total_videoinfo_list.append(repeat_inspect)
                        logger_info.info(f"Get videoinfo {index}, Continue")
                        break
            if scrapy_videoinfo_list["code"] != 0:
                failed_target += 1
                logger_info.debug(f"videoinfo is scrapy failed")
                if failed_target == 10:
                    logger_info.error(f"videoinfo is scrapy breaking")
                return index
        if index % 5 == 0:
            ## 存储爬取信息
            total_videoinfo = total_videoinfo.append(total_videoinfo_list)
            total_videoinfo.to_excel(
                total_videoinfo_add, engine="xlsxwriter", encoding="utf-8", index=False
            )
            total_videoinfo_list = pd.DataFrame()
            logger_info.success(
                f"Successfully stored videoinfo information, {len(total_videoinfo)} items total."
            )  ## 爬取信息存储成功了

    ## 存储爬取信息
    total_videoinfo = total_videoinfo.append(total_videoinfo_list)
    total_videoinfo.to_excel(
        total_videoinfo_add, engine="xlsxwriter", encoding="utf-8", index=False
    )
    total_videoinfo_list = pd.DataFrame()
    logger_info.success(
        f"Successfully stored videoinfo information, Finally {len(total_videoinfo)-total_videoinfo_num} items total."
    )  ## 爬取信息存储成功了
    for i in trace_list:
        logger.remove(i)

if __name__ == "__main__":
    main()
