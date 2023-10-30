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
    rawdata=pd.read_excel('./backup/merge_record.xlsx',dtype=str)
    rawdata=rawdata.drop_duplicates(subset='aweme_id')
    rawdata=rawdata[rawdata['comment_get'].notna()]
    rawdata['comment_count']=rawdata['comment_count'].astype(int)
    rawdata=rawdata[rawdata['comment_count']>40]
    token = getConfig("gnn_spider", "token")
    total_comment_add = getConfig("gnn_spider", "comment_address")
    total_comment_num = 0
    try:
        total_comment = pd.read_excel(total_comment_add, dtype=str)
        total_comment_num = len(total_comment)
    except:
        total_comment = pd.DataFrame(
            columns=[
                "aweme_id",
                "can_share",
                "cid",
                "create_time",
                "digg_count",
                "image_list",
                "ip_label",
                "is_author_digged",
                "is_hot",
                "is_note_comment",
                "item_comment_total",
                "label_list",
                "label_text",
                "label_type",
                "level",
                "reply_comment",
                "reply_comment_total",
                "reply_id",
                "reply_to_reply_id",
                "status",
                "stick_position",
                "text",
                "text_extra",
                "text_music_info",
                "user",
                "user_buried",
                "user_digged",
            ]
        )

    payload = {}
    headers = {"User-Agent": "Apifox/1.0.0 (https://www.apifox.cn)"}
    failed_target = 0
    trace_list = []
    trace_list.append(
        logger.add(
            "log/CommentInfo.log",
            filter=lambda record: record["extra"]["name"] == "CommentInfo",
        )
    )
    logger_info = logger.bind(name="CommentInfo")

    ## 爬取页号
    counter=0
    page=2
    total_comment_list = pd.DataFrame()
    for index, row in tqdm(rawdata.iterrows()):
        url = f"http://ttt.258data.com/dy/comment/app?video_id={row.aweme_id}&page={page}&token={token}"
        response = requests.request("GET", url, headers=headers, data=payload)
        scrapy_comment_list = response.json()
        if scrapy_comment_list["code"] == 0:
            failed_target = 0
            if scrapy_comment_list["comments"] != None and (
                len(scrapy_comment_list["comments"]) > 0
            ):
                repeat_inspect = pd.DataFrame(scrapy_comment_list["comments"])
                obtain = len(repeat_inspect)
                total_comment_list = total_comment_list.append(repeat_inspect)
                rawdata.loc[index, "comment_number"] = obtain
            else:
                rawdata.loc[index, "comment_number"] = 0

        else:  # 请求失败 返回页号
            time.sleep(1)
            for i in range(3):
                response = requests.request("GET", url, headers=headers, data=payload)
                scrapy_comment_list = response.json()
                if scrapy_comment_list["code"] == 0:
                    failed_target = 0
                    logger_info.info("Failed Retry, Success")  ## 请求返回成功了
                if scrapy_comment_list["comments"] != None and (
                    len(scrapy_comment_list["comments"]) > 0
                ):
                    repeat_inspect = pd.DataFrame(scrapy_comment_list["comments"])
                    obtain = len(repeat_inspect)
                    total_comment_list = total_comment_list.append(repeat_inspect)
                    rawdata.loc[index, "comment_number"] = obtain
                else:
                    rawdata.loc[index, "comment_number"] = 0
                break
            if scrapy_comment_list["code"] != 0:
                failed_target += 1
                logger_info.debug(f"comment is scrapy failed")
                if failed_target == 3:
                    logger_info.error(f"comment is scrapy breaking")
                sys.exit()
        logger_info.info(
            f"Get Comment Success,{len(total_comment_list)-counter} items get"
        )  ## 单次计数
        counter=len(total_comment_list)
        if index%100==0:
        ## 存储爬取信息
            total_comment = total_comment.append(total_comment_list)
            total_comment.to_excel(
                total_comment_add, engine="xlsxwriter", encoding="utf-8", index=False
            )
            rawdata.to_excel(
                video_record_add, engine="xlsxwriter", encoding="utf-8", index=False
            )
            total_comment_list = pd.DataFrame()
            counter=0
            logger_info.success(
                f"Successfully stored comment information, {len(total_comment)} items total."
            )  ## 爬取信息存储成功了
            

    ## 存储爬取信息
    total_comment = total_comment.append(total_comment_list)
    total_comment.to_excel(
        total_comment_add, engine="xlsxwriter", encoding="utf-8", index=False
    )
    rawdata.to_excel(
        video_record_add, engine="xlsxwriter", encoding="utf-8", index=False
    )
    total_comment_list = pd.DataFrame()
    logger_info.success(
        f"Successfully stored comment information, {len(total_comment)-total_comment_num} items total."
    )  ## 爬取信息存储成功了
    for i in trace_list:
        logger.remove(i)


if __name__ == "__main__":
    main()
