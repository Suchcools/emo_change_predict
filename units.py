import os
from func_timeout import func_set_timeout
import requests
from tqdm import tqdm  # 进度条模块
import configparser


# 读取配置文件
def getConfig(section, option):
    """
    :param filename 文件名称
    :param section: 服务
    :param option: 配置参数
    :return:返回配置信息
    """
    # print(configPath)

    # 创建ConfigParser对象
    conf = configparser.ConfigParser()

    # 读取文件内容
    conf.read('./config.ini',encoding='utf-8')
    config = conf.get(section, option)
    return config

@func_set_timeout(200)#设定函数超执行时间_
def down_from_url(url, dst ,logger):
    # 设置stream=True参数读取大文件
    try:
        headers = {
            'user-agent': 'Android',
        }
        response = requests.get(url, stream=True, timeout=30, headers=headers)
        # 通过header的content-length属性可以获取文件的总容量
        file_size = int(response.headers['content-length'])
        if os.path.exists(dst):
            # 获取本地已经下载的部分文件的容量，方便继续下载，如果不存在就从头开始下载。
            first_byte = os.path.getsize(dst)
        else:
            first_byte = 0
        # 如果大于或者等于则表示已经下载完成，否则继续
        if first_byte >= file_size:
            return file_size
        header = {"Range": f"bytes={first_byte}-{file_size}", 'user-agent': 'Android'}

        pbar = tqdm(total=file_size, initial=first_byte, unit='B', unit_scale=True, desc=dst)
        req = requests.get(url, headers=header, stream=True)
        with open(dst, 'ab') as f:
            # 每次读取一个1024个字节
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(1024)
        pbar.close()
        return file_size
    except:
        logger.error(f'{dst} download failed')
        return 0

@func_set_timeout(300)#设定函数超执行时间_
def download_by_addr(aweme_id,play_addr,logger):
    try:
        down_from_url(str(play_addr),f"{aweme_id}.mp4",logger)
    except:
        logger.error(f'{aweme_id} timeout failed')


