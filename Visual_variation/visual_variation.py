import cv2
import numpy as np
import pandas as pd
# 定义保存图片函数
# image:要保存的图片
# pic_address：图片保存地址
# num: 图片后缀名，用于区分图片，int 类型

# 定义了一些辅助函数：
# compare_images(img1, img2)：计算两个图像的曼哈顿范数，并返回差异值。
# normalization(data)：对数据进行归一化处理。
# normalize(arr)：使用最大-最小值范围将数组标准化为0-255之间的值。
# split_list_n_list(origin_list, n)：将原始列表分成n个子列表并返回生成器。
# 定义了video_to_pic(video_path)函数，该函数读取指定视频文件并将其拆分成多个图像。首先读取视频文件，然后使用cv2.VideoCapture()函数读取帧。接着，将每隔固定帧保存的帧记录下来。最后将每个帧转换为灰度图像，并返回拆分后的图像列表。

# 定义了get_variation(id)函数，该函数计算指定视频文件的视觉变化。它首先调用video_to_pic()函数获取图像列表。然后，它迭代图像列表并使用compare_images()函数计算当前图像与上一个图像之间的视觉差异。最后，它返回平均视觉差异值。如果处理视频文件时发生错误，它会返回字符串'NA'。


# compare_images(img1, img2)函数：这个函数将两个图像作为输入，计算它们之间的曼哈顿范数，并返回结果。它首先对两个图像进行标准化处理，以补偿它们之间的曝光差异，然后计算它们的差异并计算其曼哈顿范数。最后，它返回差异值作为输出。

# normalization(data)函数：这个函数将一个数据集作为输入，并返回对其进行归一化处理后的结果。它计算数据集的最大值和最小值，然后使用这些值将数据集标准化为0-1之间的值。

# normalize(arr)函数：这个函数将一个数组作为输入，并将其标准化为0-255之间的值。它计算数组的最大值和最小值，并使用这些值将数组标准化为0-255之间的值。

# split_list_n_list(origin_list, n)函数：这个函数将一个原始列表分成n个子列表，并返回一个生成器。它首先计算每个子列表应该包含的元素数量，然后使用生成器将原始列表分成相应数量的子列表。

# video_to_pic()的函数，它使用OpenCV库读取指定路径的视频文件，并将其拆分成多个图像。它使用cv2.VideoCapture()函数读取视频文件，并使用cv2.cvtColor()函数将读取的帧转换为灰度图像。它使用split_list_n_list()函数将每隔固定帧保存的帧分成多个子列表，并返回拆分后的图像列表。

# get_variation(id)的函数，它计算指定视频文件的视觉变化。它调用video_to_pic()函数获取图像列表，并迭代这个列表来计算每个图像与上一个图像之间的视觉差异。它使用compare_images()函数来计算视觉差异，并将它们相加以计算总体视觉差异。最后，它将总体视觉差异值除以图像数量来计算平均视觉差异值，并将其作为输出返回。如果处理视频文件时发生错误，它会返回字符串'NA'。


# 视频文件和图片保存地址
def compare_images(img1, img2):
    # normalize to compensate for exposure difference, this may be unnecessary
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = np.sum(abs(diff))  # Manhattan norm
    return m_norm
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

def split_list_n_list(origin_list, n):
    if len(origin_list) % n == 0:
        cnt = len(origin_list) // n
    else:
        cnt = len(origin_list) // n + 1
 
    for i in range(0, n):
        yield origin_list[i*cnt:(i+1)*cnt][0]

def video_to_pic(video_path):
    # 读取视频文件
    frame_count = 0
    frame_record = []
    videoCapture = cv2.VideoCapture(video_path)
    # 读帧
    success, frame = videoCapture.read()
    while success:
        frame_count = frame_count + 1
        # 每隔固定帧保存一张图片
        frame_record.append(frame)
        success, frame = videoCapture.read()
    frame_record=list(split_list_n_list(frame_record, 10))
    frame_record=[cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in frame_record]
    return frame_record

def get_variation(id):
    try:
        frame_record=video_to_pic(f'./video/{id}.mp4')
        total_manhattan_norm=0
        frame_nums=len(frame_record)
        last_frame=[]
        for i in range(frame_nums):
            # read images as 2D arrays (convert to grayscale for simplicity)
            if i==0:
                last_frame = frame_record[0]
                continue
            img = frame_record[i]
            # compare
            total_manhattan_norm += compare_images(last_frame, img)/img.size
            last_frame=img
        # print ("Visual Variation:", total_manhattan_norm/frame_nums)
        return total_manhattan_norm/frame_nums
    except:
        return 'NA'

if __name__ == '__main__':
    data=pd.read_excel('backup/intention_record.xlsx',dtype=str)
    data['variation']=data['aweme_id'].apply(get_variation)
    data.to_excel('./backup/variation_record.xlsx', engine="xlsxwriter", encoding="utf-8", index=False)