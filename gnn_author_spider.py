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
    rawdata = pd.read_excel(video_record_add,dtype=str)
    rawdata = rawdata.drop_duplicates(subset="author_user_id")
    token = getConfig("gnn_spider", "token")
    total_author_add = getConfig("gnn_spider", "author_address")
    total_author_num = 0
    try:
        total_author = pd.read_excel(total_author_add, dtype=str)
        total_author_num = len(total_author)
    except:
        total_author = pd.DataFrame(
            columns=[
                "apple_account",
                "avatar_168x168",
                "avatar_300x300",
                "avatar_larger",
                "avatar_medium",
                "avatar_thumb",
                "aweme_count",
                "aweme_count_correction_threshold",
                "birthday_hide_level",
                "can_set_item_cover",
                "can_show_group_card",
                "card_entries",
                "city",
                "close_friend_type",
                "commerce_info",
                "commerce_user_info",
                "commerce_user_level",
                "country",
                "cover_and_head_image_info",
                "cover_colour",
                "cover_url",
                "custom_verify",
                "district",
                "dongtai_count",
                "dynamic_cover",
                "enable_wish",
                "enterprise_user_info",
                "enterprise_verify_reason",
                "favorite_permission",
                "favoriting_count",
                "follow_guide",
                "follow_status",
                "follower_count",
                "follower_request_status",
                "follower_status",
                "following_count",
                "forward_count",
                "gender",
                "general_permission",
                "has_e_account_role",
                "has_subscription",
                "im_primary_role_id",
                "im_role_ids",
                "image_send_exempt",
                "ins_id",
                "ip_location",
                "is_activity_user",
                "is_ban",
                "is_block",
                "is_blocked",
                "is_effect_artist",
                "is_gov_media_vip",
                "is_mix_user",
                "is_not_show",
                "is_series_user",
                "is_sharing_profile_user",
                "is_star",
                "life_story_block",
                "live_commerce",
                "live_status",
                "max_follower_count",
                "message_chat_entry",
                "mix_count",
                "mplatform_followers_count",
                "nickname",
                "original_musician",
                "pigeon_daren_status",
                "pigeon_daren_warn_tag",
                "profile_tab_info",
                "profile_tab_type",
                "province",
                "publish_landing_tab",
                "r_fans_group_info",
                "recommend_reason_relation",
                "recommend_user_reason_source",
                "risk_notice_text",
                "room_id",
                "school_name",
                "sec_uid",
                "secret",
                "series_count",
                "share_info",
                "short_id",
                "show_favorite_list",
                "show_subscription",
                "signature",
                "signature_display_lines",
                "signature_language",
                "sync_to_toutiao",
                "tab_settings",
                "total_favorited",
                "total_favorited_correction_threshold",
                "twitter_id",
                "twitter_name",
                "uid",
                "unique_id",
                "urge_detail",
                "user_age",
                "user_not_see",
                "user_not_show",
                "verification_type",
                "video_cover",
                "video_icon",
                "watch_status",
                "white_cover_url",
                "with_commerce_enterprise_tab_entry",
                "with_commerce_entry",
                "with_fusion_shop_entry",
                "with_new_goods",
                "youtube_channel_id",
                "youtube_channel_title",
            ]
        )

    payload = {}
    headers = {"User-Agent": "Apifox/1.0.0 (https://www.apifox.cn)"}
    failed_target = 0
    trace_list = []
    trace_list.append(
        logger.add(
            "log/AuthorInfo.log",
            filter=lambda record: record["extra"]["name"] == "AuthorInfo",
        )
    )
    logger_info = logger.bind(name="AuthorInfo")

    ## 爬取页号
    total_author_list = pd.DataFrame()
    for index, row in tqdm(rawdata.iterrows()):
        url = (
            f"http://ttt.258data.com/dy/user/uid?uid={row.author_user_id}&token={token}"
        )
        response = requests.request("GET", url, headers=headers, data=payload)
        scrapy_author_list = response.json()
        if scrapy_author_list["code"] == 0:
            failed_target = 0
            if scrapy_author_list["user"] != None:
                repeat_inspect = pd.DataFrame([scrapy_author_list["user"]])
                total_author_list = total_author_list.append(repeat_inspect)

        else:  # 请求失败 返回页号
            time.sleep(1)
            for i in range(3):
                response = requests.request("GET", url, headers=headers, data=payload)
                scrapy_author_list = response.json()
                if scrapy_author_list["code"] == 0:
                    failed_target = 0
                    logger_info.info("Failed Retry, Success")  ## 请求返回成功了
                if scrapy_author_list["user"] != None:
                    repeat_inspect = pd.DataFrame([scrapy_author_list["user"]])
                    total_author_list = total_author_list.append(repeat_inspect)
                break
            if scrapy_author_list["code"] != 0:
                failed_target += 1
                logger_info.debug(f"author is scrapy failed")
                if failed_target == 3:
                    logger_info.error(f"author is scrapy breaking")
                sys.exit()
        if index % 100 == 0:
            ## 存储爬取信息
            total_author = total_author.append(total_author_list)
            total_author.to_excel(
                total_author_add, engine="xlsxwriter", encoding="utf-8", index=False
            )
            total_author_list = pd.DataFrame()
            logger_info.success(
                f"Successfully stored author information, {len(total_author)} items total."
            )  ## 爬取信息存储成功了

    ## 存储爬取信息
    total_author = total_author.append(total_author_list)
    total_author.to_excel(
        total_author_add, engine="xlsxwriter", encoding="utf-8", index=False
    )
    total_author_list = pd.DataFrame()
    logger_info.success(
        f"Successfully stored author information, Finally {len(total_author)-total_author_num} items total."
    )  ## 爬取信息存储成功了
    for i in trace_list:
        logger.remove(i)

if __name__ == "__main__":
    main()
