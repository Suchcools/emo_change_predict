## 检查一次
from units import download_by_addr,getConfig
import pandas as pd
from loguru import logger
address = getConfig("gnn_spider", "address")

logger.add(
    "log/VideoInspectRepeat.log",
)

info=pd.read_excel(address,dtype=str)
info.apply(lambda x:download_by_addr('./video/'+str(x.aweme_id),x.play_addr,logger),axis=1)