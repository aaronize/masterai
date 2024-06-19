"""

"""
import json
import time
from datetime import datetime
from typing import List, Any
from http import HTTPStatus

import requests
from bs4 import BeautifulSoup
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Message, Role
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool, BaseTool
from operator import itemgetter

from src.utils.env import QWEN_API_KEY, CALENDAR_API_KEY, WEATHER_API_KEY

API_KEY = QWEN_API_KEY
MAX_RETRY_TIMES = 2


def call_with_stream():
    messages = [Message(role='user', content='如何做西红柿炖牛腩？')]

    responses = Generation.call("qwen-max",
                                api_key=API_KEY,
                                messages=messages,
                                result_format='message',  # 设置输出为'message'格式
                                stream=True,  # 设置输出方式为流式输出
                                incremental_output=True  # 增量式流式输出
                                )
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            print(response.output.choices[0]['message']['content'], end='')
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))


class ReactQwen(object):
    def __init__(self, tool_list):
        self.tool_list = tool_list
        self.tools = [convert_to_openai_tool(t) for t in tool_list]
        print(">>>> tools:", self.tools)
        self.tool_name = [t["function"]["name"] for t in self.tools]
        print(">>>> function names:", self.tool_name)
        self.parser = JsonOutputParser()

    def prompt_qwen(self, content: str) -> List[Message]:
        """

        :param content:
        :return:
        """
        sys_prompt_t = f'''Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take in order, should be one of {self.tool_name}
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {content}
Thought:'''
        prompt = [
            Message(role=Role.SYSTEM,
                    content='Answer the following questions as best you can. You have access to the following tools'),
            Message(role=Role.USER, content=sys_prompt_t)
        ]

        return prompt

    def get_response(self, messages: List[Message]):
        """
        调用qwen大模型，获取输入结果
        :param messages:
        :return:
        """
        retry_times = 0
        err_msg = None

        while True:
            if retry_times > MAX_RETRY_TIMES:
                raise Exception(f"Call error {err_msg}")

            res = Generation.call(
                "qwen-max",
                api_key=API_KEY,
                messages=messages,
                result_format="message",
                tools=self.tools,
            )
            if res.status_code == 200:
                return res
            else:
                err_msg = res.message
                # retry times
                retry_times += 1
                print(f"[ERR] call qwen error, retry in 10s. err: {res.message}")
                time.sleep(10)

    def parse_content(self, out_content):
        """

        :param out_content:
        :return:
        """
        return {
            "name": out_content.split("\nAction: ")[1].split("\nAction")[0],
            "arguments": self.parser.parse(out_content.split("Input: ")[1])
        }

    def tool_chain(self, model_output):
        """

        :param model_output:
        :return:
        """
        tool_map = {t.name: t for t in self.tool_list}
        chosen_tool = tool_map[model_output["name"]]

        # 这里itemgetter("arguments")表示在tool_chain.invoke(args)时，取出并返回args的arguments字段的值
        return itemgetter("arguments") | chosen_tool

    def invoke(self, input_p):
        """

        :param input_p:
        :return:
        """
        prompt = self.prompt_qwen(input_p)

        for i in range(0, 5):
            # print(f"第[{i}]次，Prompt：", prompt)
            res = self.get_response(prompt)
            res_content = res.output.choices[0].message["content"]
            # print(">>> res content:", res_content)

            if res_content.find("\nAction: ") != -1:
                tool_args = self.parse_content(res_content)
                tool_out = self.tool_chain(tool_args)
                print(">>> tool out:", tool_out)
                prompt[1].content = prompt[1].content + res_content + "\nObservation: " + str(
                    tool_out.invoke(tool_args)) + "\nThought:"
            else:
                prompt[1].content = prompt[1].content + res_content
                break

        return prompt[1].content


@tool
def multiply(first_int: int, second_int: int) -> int:
    """将两个整数相乘。"""
    return first_int * second_int


@tool
def add(first_int: int, second_int: int) -> int:
    """将两个整数相加。"""
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    """对底数求指数幂。"""
    return base ** exponent


@tool
def weather(city: str) -> dict:
    """查询最近几日的天气情况，包括温度，天气，湿度，风向等"""
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    url = f"http://apis.juhe.cn/simpleWeather/query"
    params = {
        "key": WEATHER_API_KEY,
        "city": city
    }
    res = requests.get(url, params, headers=headers, timeout=5)
    res_data = res.json()

    return res_data


@tool
def calendar(date: str) -> dict:
    """查询指定日期（日期格式如：'2024-04-08'）的信息，包括农历，星期几，假期，生肖，习俗，忌讳等"""
    date = date.replace("-0", "-")

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    url = f"http://v.juhe.cn/calendar/day"
    params = {
        "key": CALENDAR_API_KEY,
        "date": date
    }
    res = requests.get(url, params, headers=headers, timeout=5)
    res_data = res.json()

    return res_data


@tool
def get_date() -> str:
    """获取当天日期"""
    return datetime.now().date().strftime("%Y-%m-%d")


@tool
def get_year() -> str:
    """获取年份"""
    return str(datetime.now().year)


@tool
def baidu_search(keyword: str) -> str:
    """通过搜索引擎查询指定关键词的相关资料"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.baidu.com",
        "Cookie": "BIDUPSID=F8196EE710E002BEBB8A5AB2BF33C8FE; PSTM=1711510185; BD_UPN=123253; BAIDUID=123F03F019BD80F0B118A6504356A647:FG=1; BAIDUID_BFESS=123F03F019BD80F0B118A6504356A647:FG=1; ZFY=:BatzCEVBTH82Zebbpi2HNlDP:AAFB5Be3OujB3sQeUnM:C; H_WISE_SIDS=60298_60253_60316_60327; BDUSS=FzVlNpbXV6Ry11dmZDRzgtOHZvakt4c3hFdXl2ejl5dlh1dHAtcTZzWFEzNUZtSUFBQUFBJCQAAAAAAAAAAAEAAAAfx~8Eb25seaLegaWpZwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANBSambQUmpmSn; BDUSS_BFESS=FzVlNpbXV6Ry11dmZDRzgtOHZvakt4c3hFdXl2ejl5dlh1dHAtcTZzWFEzNUZtSUFBQUFBJCQAAAAAAAAAAAEAAAAfx~8Eb25seaLegaWpZwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANBSambQUmpmSn; bce-sessionid=001c1899c87781f4fea86242f55a4dcb73b; ab_sr=1.0.1_ODVlNGYwNmFjYzNkYjE1ZjQ5ZWEyNTMxNDU5NzQ0YmQwMDY0NTAwMzhhMWNmNjRhY2IxNWM2Y2IzOTU0MDVkN2NkZDQ2YzdhZTYwYzE0ZWYwNmFiMzY5ODdjZDZmZGIyODQ2NzEwNjk0Y2MzNDQ0Mjc0ZWRkYTkxZTgzN2U5MDgyMTg0YTJiMzM2NjZiNjNiYTU3YzllMDc4MjMzM2RhMTM0YTljODExMGUzZTM2MjUyNWRmNTI4Y2E1NjFlNjc1; H_PS_PSSID=60327_60340_60297; BA_HECTOR=8ka401ah85ak05akal24ah2l3npbsf1j6kpnh1u; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; BDRCVFR[feWj1Vr5u3D]=I67x6TjHwwYf0; BD_CK_SAM=1; PSINO=3; delPer=0; H_PS_645EC=d701K52XrcHEiBAg19mRwPQsnVLKGawXBcRU%2FdGsAdkdB2HHazE2zvtW5MMYiEErLG3X; BDSVRTM=188; COOKIE_SESSION=1046_0_4_4_10_18_1_0_3_4_1_3_5679805_0_24_0_1717637039_0_1717637015%7C4%230_0_1718249229%7C1%7C1; WWW_ST=1718249254785"
    }
    url = "https://www.baidu.com/s?wd=" + keyword
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text)

    text = ""
    for i in soup.find_all("div", class_="result c-container xpath-log new-pmd"):
        try:
            text = text + "title:" + i.find("a").text + "\n"
            text = text + "content:" + i.find("span", class_="content-right_2s-H4").text + "\n"
        except:
            continue

    return text


@tool
def toutiao_search(keyword: str) -> str:
    """通过新闻搜索查询指定关键词的最新资讯"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Accept-Encoding": "gzip, deflate",
        "Host": "so.toutiao.com",
        "Cookie": "tt_webid=7352705282449819176; _ga=GA1.1.926321135.1711935120; msToken=tZCXTCq5TzFs83FnEk0lNH5-gy4h61R_mY8U67At8vs5FFCw7FGMEMjsqIxStJ0LRsWMc8f9i0_Y5HmDo34fs09Hb1rTLaS8jIX0B-uo; _ga_QEHZPBE5HH=GS1.1.1718250323.3.0.1718250323.0.0.0; ttwid=1%7COGocwLgIPjDw7IMy00EfBkM4UpR3RWNfP6E_l1o1t_8%7C1718250324%7Cb9a72ff5f3509d72a620248cd8adb6ad0f84e674ea5cce2030267d424abfff33; _tea_utm_cache_4916=undefined; _S_WIN_WH=1792_950; _S_DPR=2; _S_IPAD=0; s_v_web_id=verify_lxcpwhzv_kKiWn686_CNfz_4dt4_8Uux_40wP2uQNLaPB"
    }
    url = f"https://so.toutiao.com/search?dvpf=pc&source=input&keyword={keyword}&page_num=0&pd=synthesis"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text)

    text = ""
    for i in soup.find_all("div", class_="result-content")[1:11]:
        try:
            text = text + "title:" + i.find("a").text + "\n"
            text = text + "content:" + i.find("span", class_="text-underline-hover").text + "\n"
        except:
            continue

    return text


class GoogleSearchTool(BaseTool):
    name = ""
    description = ""

    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        pass

    def _arun(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("This tool dose not support async run.")


if __name__ == '__main__':
    # call_with_stream()
    # 简单算术计算
    # tools = [multiply, add, exponentiate]
    # calculator = ReactQwen(tool_list=tools)
    # res = calculator.invoke("5加7,然后乘以6,然后再求结果的2次幂")
    # print(res)

    # 日历和天气接口实现出行顾问
    # res = weather.invoke("上海")
    # print(res)
    # date = get_date()
    # print(date)
    # res = calendar.invoke({"date": "2024-06-13"})
    # print(res)
    #
    tools = [get_date, weather, calendar]
    traveling = ReactQwen(tool_list=tools)
    res = traveling.invoke("后天上海适合出行吗？有什么忌讳吗？")
    print(res)

    #
    # res = search_engine("今年上海高考语文作文题目是什么？")
    # print(res)
    # res = toutiao_search("今年上海高考语文作文题目是什么？")
    # print(res)
    # tools = [get_date, baidu_search, toutiao_search]
    # searching = ReactQwen(tool_list=tools)
    # # res = searching.invoke("今年上海高考语文作文题目是什么？")
    # res = searching.invoke("今年上海高考语文作文题目是什么？")
    # print(res)
