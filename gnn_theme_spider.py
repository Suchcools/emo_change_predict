import requests
import pandas as pd
import sys
from units import download_by_addr, getConfig
import warnings
warnings.filterwarnings("ignore")
from loguru import logger


def main(keyword):
    # 配置信息
    start = int(getConfig("gnn_spider", "start"))
    end = int(getConfig("gnn_spider", "end"))
    video_save_add = getConfig("gnn_spider", "video_address")
    token = getConfig("gnn_spider", "token")
    total_info_add = getConfig("gnn_spider", "record_address")
    theme = getConfig("gnn_spider", "theme")

    try:
        total_info = pd.read_excel(total_info_add, dtype=str)
    except:
        total_info = pd.DataFrame(
            columns=[
                "author",
                "author_user_id",
                "aweme_id",
                "aweme_type",
                "cha_list",
                "comment_list",
                "create_time",
                "desc",
                "duration",
                "forward_id",
                "geofencing",
                "group_id",
                "group_id_str",
                "image_infos",
                "images",
                "is_live_replay",
                "is_preview",
                "label_top_text",
                "long_video",
                "music",
                "promotions",
                "share_info",
                "share_url",
                "text_extra",
                "video",
                "video_labels",
                "video_text",
                "anchor_info",
                "aweme_poi_info",
                "category",
                "from_xigua",
                "keyword",
            ]
        )

    payload = {}
    headers = {"User-Agent": "Apifox/1.0.0 (https://www.apifox.cn)"}
    failed_target = 0
    trace_list = []
    trace_list.append(
        logger.add(
            "log/VideoInfo.log",
            filter=lambda record: record["extra"]["name"] == "VideoInfo",
        )
    )

    trace_list.append(
        logger.add(
            "log/VideoDownload.log",
            filter=lambda record: record["extra"]["name"] == "VideoDownload",
        )
    )

    trace_list.append(
        logger.add(
            "log/VideoInspect.log",
            filter=lambda record: record["extra"]["name"] == "VideoInspect",
        )
    )

    logger_info = logger.bind(name="VideoInfo")
    logger_download = logger.bind(name="VideoDownload")
    logger_inspect = logger.bind(name="VideoInspect")

    ## 爬取页号
    for page in range(start, end):
        total_video_list = pd.DataFrame()
        url = f"http://ttt.258data.com/dy/video/topic/list?ch_id={theme}&cursor={page}&token={token}"
        response = requests.request("GET", url, headers=headers, data=payload)
        scrapy_vedio_list = response.json()
        if scrapy_vedio_list["code"] == 0 and (len(scrapy_vedio_list["aweme_list"])>0):
            failed_target = 0
            logger_info.info("Get Info Success")  ## 请求返回成功了
            repeat_inspect = pd.DataFrame(scrapy_vedio_list["aweme_list"])
            repeat_inspect = repeat_inspect[
                ~repeat_inspect.aweme_id.isin(total_info.aweme_id)
            ]
            obtain = len(repeat_inspect)
            total_video_list = total_video_list.append(repeat_inspect)

        else:  # 请求失败 返回页号
            for i in range(3):
                response = requests.request("GET", url, headers=headers, data=payload)
                scrapy_vedio_list = response.json()
                if scrapy_vedio_list["code"] == 0 and (len(scrapy_vedio_list["aweme_list"])>0):
                    failed_target = 0
                    logger_info.info("Failed Retry, Success")  ## 请求返回成功了

                    repeat_inspect = pd.DataFrame(scrapy_vedio_list["aweme_list"])
                    repeat_inspect = repeat_inspect[
                        ~repeat_inspect.aweme_id.isin(total_info.aweme_id)
                    ]
                    obtain = len(repeat_inspect)
                    total_video_list = total_video_list.append(repeat_inspect)
                    continue
            if scrapy_vedio_list["code"] != 0 or (len(scrapy_vedio_list["aweme_list"])==0):
                failed_target += 1
                logger_info.debug(
                    f"Page {page} is scrapy failed , output : has_more {scrapy_vedio_list['has_more']}"
                )
                if failed_target == 3:
                    logger_info.error(f"Page {page} is scrapy breaking")
                    if len(total_video_list) == 0:
                        logger_info.critical(f"Interface error, no data obtained")
                        sys.exit()
                    break

        repeat_inspect.video = repeat_inspect.video.apply(lambda x: eval(str(x)))

        ## 存储爬取信息
        info = total_video_list.drop_duplicates(subset=["aweme_id"], keep="first")
        info["keyword"] = keyword
        info.video = info.video.apply(lambda x: eval(str(x)))
        total_info = total_info.append(info)
        total_info.video = total_info.video.apply(lambda x: eval(str(x)))
        total_info.to_excel(
            total_info_add, engine="xlsxwriter", encoding="utf-8", index=False
        )


        ### 部分下载
        try:
            if not repeat_inspect.empty:
                repeat_inspect.apply(
                    lambda x: download_by_addr(
                        video_save_add + str(x.aweme_id),
                        x.video["play_addr"]["url_list"][0].replace("/playwm/", "/play/"),
                        logger_download,
                    ),
                    axis=1,
                )
                logger_info.info(f"Page {page + 1}, {obtain} Video Download Over")  ## 下载成功了
        except:
            logger_info.info(f"------------------------------Time Runout ! ------------------------------"
    ) 

    ## 存储爬取信息
    info = total_video_list.drop_duplicates(subset=["aweme_id"], keep="first")
    info["keyword"] = keyword

    info.video = info.video.apply(lambda x: eval(str(x)))
    total_info = total_info.append(info)
    total_info.video = total_info.video.apply(lambda x: eval(str(x)))
    total_info.to_excel(
        total_info_add, engine="xlsxwriter", encoding="utf-8", index=False
    )
    logger_info.success(
        f"Successfully stored crawl information, {len(info)} items get, {len(total_info)} items total."
    )  ## 爬取信息存储成功了

    ## 检查一次
    logger_info.info(f"Download Inspect")  ## 爬取信息存储成功了
    try:
        info.apply(
            lambda x: download_by_addr(
                video_save_add + str(x.aweme_id),
                x.video["play_addr"]["url_list"][0].replace("/playwm/", "/play/"),
                logger_inspect,
            ),
            axis=1,
        )
    except:
         logger_info.info(
        f"------------------------------ Time Runout ! ------------------------------"
    ) 

    logger_info.info(
        f"------------------------------ {keyword} Over ! ------------------------------"
    )  ## 爬取信息存储成功了
    for i in trace_list:
        logger.remove(i)


if __name__ == "__main__":
    # 健身vlog 
    keyword_list = ["健身vlog"]

    for keyword in keyword_list:
        main(keyword)
