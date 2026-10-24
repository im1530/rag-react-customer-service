from langchain.agents import create_agent

from model.factory import chat_model
from utils.prompts_loader import load_system_prompt
from agent.tools.agent_tools import (rag_summarize,get_weather,get_user_id,get_user_location,get_current_month,
                                     fetch_external_data,fill_context_for_report)
from agent.tools.middleware import (log_before_model,monitor_tool,report_prompt_switch)




class ReactAgent:
    """ReAct 智能体：组装模型、工具、中间件，并以流式方式把回答逐段吐给前端。"""
    # 初始化智能体
    def __init__(self):
        self.agent = create_agent(
            model = chat_model,                                                                     # 聊天大模型
            system_prompt = load_system_prompt(),                                                   # 默认主客服提示词
            tools = [rag_summarize,get_weather,get_user_id,get_user_location,get_current_month,     # 模型可调用的全部工具
                    fetch_external_data,fill_context_for_report],
            middleware = [log_before_model,monitor_tool,report_prompt_switch]                       # 请求处理中间件
        )

    def execute_stream(self,query: str):
        """
        以流式方式运行 Agent，逐段 yield 模型最新消息的文本内容。
            :param query: 用户输入的问题
            :return: 生成器，逐段产出回答文本
        """
        # 组装 LangGraph 输入：用户消息放进 messages 列表
        input_dict = {
            "messages":[
                {"role": "user", "content": query}
            ]
        }

        # stream 流式运行，context 初始化报告标志位为 False；values 模式每次返回完整状态
        for chunk in self.agent.stream(
            input = input_dict,
            stream_mode = "values",
            context = {"report":False}
        ):
            latest_message = chunk['messages'][-1]              # 取本轮最新一条消息
            if latest_message.content:                          # 有文本内容才输出（工具调用等可能为空）
                yield latest_message.content.strip() + "\n"     # 逐段吐出，末尾换行



# 测试
if __name__ == "__main__":
    agent = ReactAgent()
    for chunk in agent.execute_stream("扫地机器在我所在的地区的气温下如何保养"):
        print(chunk, end="",flush=True)
