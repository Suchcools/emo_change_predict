from loguru import logger
import requests
import pandas as pd
import sys
from units import download_by_addr, getConfig
import warnings
warnings.filterwarnings("ignore")

# 配置信息
start = int(getConfig("gnn_spider", "start"))
end = int(getConfig("gnn_spider", "end"))
keyword = getConfig("gnn_spider", "keyword")
token = getConfig("gnn_spider", "token")
total_info_add = getConfig("gnn_spider", "address")

try:
    total_info = pd.read_excel(total_info_add, dtype=str)
except:
    total_info = pd.DataFrame(columns=['aweme_id', 'collect_count', 'comment_count', 'create_time', 'desc',
                                       'digg_count', 'download_count', 'duration', 'follower_count', 'gender',
                                       'location', 'nickname', 'play_addr', 'poi_info', 'region', 'sec_uid',
                                       'share_count', 'share_url', 'short_id', 'signature', 'uid', 'unique_id',
                                       'user_image_url', 'vedio_img', 'video_tag', 'keyword'])

payload = {}
headers = {"User-Agent": "Apifox/1.0.0 (https://www.apifox.cn)"}
total_video_list = pd.DataFrame()
failed_target = 0

logger.add(
    "log/VideoInfo.log", filter=lambda record: record["extra"]["name"] == "VideoInfo"
)
logger.add(
    "log/VideoDownload.log",
    filter=lambda record: record["extra"]["name"] == "VideoDownload",
)
logger.add(
    "log/VideoInspect.log",
    filter=lambda record: record["extra"]["name"] == "VideoInspect",
)

logger_info = logger.bind(name="VideoInfo")
logger_download = logger.bind(name="VideoDownload")
logger_inspect = logger.bind(name="VideoInspect")

## 爬取页号
for page in range(start, end):
    url = f"http://ttt.258data.com/dy/search/video/app/v3?keyword={keyword}&page={page}&token={token}"
    response = requests.request("GET", url, headers=headers, data=payload)
    scrapy_vedio_list = response.json()
    if scrapy_vedio_list["code"] == 0:
        failed_target = 0
        logger_info.info("Get Info Success")  ## 请求返回成功了

        repeat_inspect = pd.DataFrame(scrapy_vedio_list["data"]["aweme_msg_list"])
        repeat_inspect = repeat_inspect[~repeat_inspect.aweme_id.isin(total_info.aweme_id)]
        obtain = len(repeat_inspect)
        total_video_list = total_video_list.append(repeat_inspect)

    else:  # 请求失败 返回页号
        for i in range(3):
            response = requests.request("GET", url, headers=headers, data=payload)
            scrapy_vedio_list = response.json()
            if scrapy_vedio_list["code"] == 0:
                failed_target = 0
                logger_info.info("Failed Retry, Success")  ## 请求返回成功了

                repeat_inspect = pd.DataFrame(scrapy_vedio_list["data"]["aweme_msg_list"])
                repeat_inspect = repeat_inspect[~repeat_inspect.aweme_id.isin(total_info.aweme_id)]
                obtain = len(repeat_inspect)
                total_video_list = total_video_list.append(repeat_inspect)

                continue
        if scrapy_vedio_list["code"] != 0:
            failed_target += 1
            logger_info.debug(
                f"Page {page} is scrapy failed , output : {scrapy_vedio_list['msg']}"
            )
            if failed_target == 3:
                logger_info.error(f"Page {page} is scrapy breaking")
                if len(total_video_list) == 0:
                    logger_info.critical(f"Interface error, no data obtained")
                    sys.exit()
                break
    ### 部分下载
    if not repeat_inspect.empty:
        repeat_inspect.apply(
            lambda x: download_by_addr(
                "./video/" + str(x.aweme_id), x.play_addr, logger_download
            ),
            axis=1,
        )
        logger_info.info(f"Page {page + 1}, {obtain} Video Download Over")  ## 下载成功了

## 存储爬取信息
info = total_video_list.drop_duplicates(subset=["aweme_id"], keep="first")
info['keyword'] = keyword

total_info = total_info.append(info)
total_info.to_excel(total_info_add, engine='xlsxwriter', encoding='utf-8', index=False)
logger_info.success(
    f"Successfully stored crawl information, {len(info)} items get."
)  ## 爬取信息存储成功了

## 检查一次
logger_info.info(
    f"Download Inspect"
)  ## 爬取信息存储成功了

info.apply(
    lambda x: download_by_addr(
        "./video/" + str(x.aweme_id), x.play_addr, logger_inspect
    ),
    axis=1,
)
