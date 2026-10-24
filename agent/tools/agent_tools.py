"""
Agent 工具集模块（@tool 工具封装）

把 Agent 可调用的外部能力用 @tool 装饰器包装成工具，每个工具的 description 供大模型,判断"何时调用、传什么参数"。

本模块包含两类工具：
一、演示型工具（返回写死/随机假数据，用于跑通调用流程）：
    - get_weather：返回指定城市的模拟天气；
    - get_user_location：随机返回一个模拟所在城市；
    - get_user_id：从 user_ids 中随机返回一个模拟用户ID；
    - get_current_month：从 month_arr 中随机返回一个模拟月份。

二、真实数据工具：
    - get_external_data：读取外部 CSV 数据文件，解析并缓存为两级字典
      {user_id: {month: {特征, 效率, 消耗, 对比}}}；
    - fetch_external_data：供 Agent 按 用户ID + 月份 查询使用记录，未命中返回空字符串。
"""


from langchain_core.tools import tool
import random
from rag.rag_service import RagSummarizeService
from utils.path_tool import get_abs_path
import os
from utils.config_handler import agent_conf
from utils.logger_handler import logger



# RAG 服务实例
rag = RagSummarizeService()

# 工具：进行RAG检索
@tool(description='从向量存储中检索参考资料')
def rag_summarize(question: str) -> str:
    return rag.rag_summarize(question)


# 工具：获取指定城市天气（演示用的固定假数据）
@tool(description='获取指定城市的天气,以消息字符串的形式返回')
def get_weather(city: str) -> str:
    return f"{city}的天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"


# 工具：获取用户所在城市名称（演示用的固定假数据）
@tool(description='获取用户所在城市的名称，以纯字符串形式返回')
def get_user_location() -> str:
    return random.choice(["深圳", "合肥", "杭州"])


# 用户ID列表
user_ids = ['1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009', '1010']

# 工具：随机返回一个用户ID（演示用的固定假数据）
@tool(description='获取用户的ID，以纯字符串形式返回')
def get_user_id() -> str:
    return random.choice(user_ids)


# 月份列表
month_arr = ['2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06',
             '2025-07', '2025-08', '2025-09', '2025-10', '2025-11', '2025-12']

# 工具：随机返回一个月份（演示用的固定假数据）
@tool(description='获取当前月份，以纯字符串形式返回')
def get_current_month() -> str:
    return random.choice(month_arr)


# 外部数据缓存：结构 {user_id: {month: {特征,效率,消耗,对比}}}
external_data = {}

# 函数：从外部 CSV 文件中读取数据并填充到全局 external_data 字典
def get_external_data():
    """
    读取外部 CSV 数据文件，解析后填充到全局 external_data 字典
    {
        "user_id": {
            "month": {"特征":xxx, "效率":xxx, ...}
            "month": {"特征":xxx, "效率":xxx, ...}
            "month": {"特征":xxx, "效率":xxx, ...}
            ...
        },
        "user_id": {
            "month": {"特征":xxx, "效率":xxx, ...}
            "month": {"特征":xxx, "效率":xxx, ...}
            "month": {"特征":xxx, "效率":xxx, ...}
            ...
        },
        "user_id": {
            "month": {"特征":xxx, "效率":xxx, ...}
            "month": {"特征":xxx, "效率":xxx, ...}
            "month": {"特征":xxx, "效率":xxx, ...}
            ...
        },
        ...
    }
    """
    global external_data  # 声明使用全局字典，否则下面赋值会被当成局部变量

    # 缓存为空才读文件，避免每次都重复读盘
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        # 文件不存在直接抛错
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:]:                      # readlines()[1:] 跳过第1行表头，从第2行数据开始
                data_arr: list[str] = line.strip().split(',')   # 按逗号切成字段

                user_id: str = data_arr[0].replace('"', '')          # 第0列 用户ID，去掉引号
                feature: str = data_arr[1].replace('"', '')          # 第1列 特征
                efficiency: str = data_arr[2].replace('"', '')       # 第2列 效率
                consumables: str = data_arr[3].replace('"', '')      # 第3列 消耗
                comparison : str = data_arr[4].replace('"', '')      # 第4列 对比
                time : str = data_arr[5].replace('"', '')            # 第5列 月份/时间

                # 该用户第一次出现时，先建一个空字典
                if user_id not in external_data:
                    external_data[user_id] = {}

                # 按月份存入该用户的记录
                external_data[user_id][time] = {
                    "特征":feature,
                    "效率":efficiency,
                    "消耗":consumables,
                    "对比":comparison
                }


# 工具：按 用户ID + 月份 从外部数据里取使用记录，找不到返回空字符串
@tool(description='从外部系统中获取指定用户在指定月份的使用记录,以纯字符串形式返回,如果未检索到则返回空字符串')
def fetch_external_data(user_id: str, month: str) -> str:
    get_external_data()     # 确保外部数据已加载到缓存

    try:
        # 两层字典取值：先用户，再月份
        return str(external_data[user_id][month])
    except KeyError:
        # 该用户或该月份没记录 → 记警告并返回空串
        logger.warning(f"[fetch_external_data]未能检索到用户:{user_id}在{month}的使用记录")
        return ""


@tool(description='无入参,无返回值,调用后触发中间件自动为报告生成的场景动态注入上下文信息,为后续提示词切换提供上下文信息')
def fill_context_for_report():
    # 信号型工具：本身不查数据、不算结果，作用只是"被模型调用一下"作为触发信号
    # 返回一句占位提示串（@tool 要求有返回内容），真正动作在 monitor_tool 中间件里完成
    return "fill_context_for_report已调用"



# 测试
if __name__ == '__main__':
    print(fetch_external_data("1001", "2025-01"))
