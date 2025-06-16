import random

import requests
from datetime import datetime
import urllib.parse
import time

class GuessWordAPI:
    """
    用于与猜盐(原炒饭小测验)网站交互的爬虫类，
    可以发送词汇猜测并获取相似度得分
    """

    def __init__(self, sleep_time, sleep_time_bias, random_sleep = False, base_url="https://xiaoce.fun"):
        """
        初始化爬虫类

        参数:
        base_url (str): 网站的基础URL
        """
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/v0/quiz/daily/GuessWord/guess"
        self.session = requests.Session()

        self.base_sleep_time = sleep_time
        self.random_sleep = random_sleep
        self.random_sleep_bias = sleep_time_bias

    def request_sleep(self):
        """每次请求后固定休眠 base±scale 时间"""
        if self.random_sleep:
            # 生成带随机波动的休眠时间
            sleep_time = random.uniform(
                max(.0, self.base_sleep_time - self.random_sleep_bias),
                self.base_sleep_time + self.random_sleep_bias
            )
        else:
            sleep_time = self.base_sleep_time

        time.sleep(sleep_time)  # 执行休眠

    def guess_word(self, word, date=None):
        """
        提交一个词汇并获取其与目标词的相似度

        参数:
        word (str): 要猜测的中文词汇
        date (str, optional): 日期字符串，格式为YYYYMMDD。默认为当前日期。

        返回:
        dict: 包含相似度得分的字典，格式为：
            {
                'word': '猜测的词',
                'similarity': 相似度百分比(0-100),
                'correct': 是否正确,
                'raw_response': 原始API响应
            }
        """

        # 添加请求间隔
        self.request_sleep()

        # 如果未提供日期，使用当前日期
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        # 对中文词汇进行URL编码
        encoded_word = urllib.parse.quote(word)

        # 构建API请求URL
        url = f"{self.api_endpoint}?date={date}&word={encoded_word}"

        try:
            # 发送请求
            response = self.session.get(url)
            response.raise_for_status()  # 检查是否有HTTP错误
            # 解析JSON响应
            data = response.json()

            return {
                'word': word,
                'similarity': data.get('doubleScore'),
                'correct': data.get('correct', False),
                'raw_response': data
            }

        except requests.exceptions.RequestException as e:
            print(f"请求出错: {e}")
            return {
                'word': word,
                'similarity': 0,
                'correct': False,
                'error': str(e)
            }


# 示例用法
if __name__ == "__main__":
    # 创建爬虫实例
    crawler = GuessWordAPI()

    # 猜测单个词
    result = crawler.guess_word("完美")
    print(f"词汇 '{result['word']}' 的相似度: {result['similarity']}%")