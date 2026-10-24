"""
使用yaml处理配置文件，获取配置信息
    .yml文件的格式 --- key: value(冒号后加一个空格)
"""


import yaml
from utils.path_tool import get_abs_path


# 函数:读取 RAG 配置
def get_rag_config(config_path:str = get_abs_path("config/rag.yml"),encoding = "utf-8"):
    """
    读取 RAG 相关配置（如检索参数、分块大小等）
        :param config_path: 配置文件路径，默认通过 get_abs_path 把相对路径 config/rag.yml.yml 转成项目绝对路径
        :param encoding: 文件编码，默认 utf-8，防中文乱码
        :return: 解析后的配置字典
    """
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)  # 用 FullLoader 安全解析 yml，返回成 Python 字典


# 函数:读取 Agent 配置
def get_agent_config(config_path:str = get_abs_path("config/agent.yml"),encoding = "utf-8"):
    """
    读取 Agent 相关配置（如模型名、工具、system_prompt 等）
        :param config_path: 配置文件路径，默认 config/agent.yml.yml 的绝对路径
        :param encoding: 文件编码，默认 utf-8
        :return: 解析后的配置字典
    """
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


# 函数:读取提示词配置
def get_prompts_config(config_path:str = get_abs_path("config/prompts.yml"),encoding = "utf-8"):
    """
    读取提示词（Prompt）配置，存放各类模板话术
        :param config_path: 配置文件路径，默认 config/prompts.yml.yml 的绝对路径
        :param encoding: 文件编码，默认 utf-8
        :return: 解析后的配置字典
    """
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


# 函数:读取向量库 Chroma 配置
def get_chroma_config(config_path:str = get_abs_path("config/chroma.yml"),encoding = "utf-8"):
    """
    读取向量库 Chroma 的配置（如持久化目录、集合名等）
        :param config_path: 配置文件路径，默认 config/chroma.yml 的绝对路径
        :param encoding: 文件编码，默认 utf-8
        :return: 解析后的配置字典
    """
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)



# 创建变量 - 配置信息
rag_conf = get_rag_config()
agent_conf = get_agent_config()
prompts_conf = get_prompts_config()
chroma_conf = get_chroma_config()




# 测试
if __name__ == "__main__":
    print(agent_conf["chat_model_name"])

