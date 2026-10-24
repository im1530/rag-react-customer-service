"""
Agent 中间件模块（Middleware）

通过 LangChain/LangGraph 提供的生命周期钩子，在 Agent 运行流程中插入横切逻辑（日志监控、状态标记、动态提示词切换），而不改动工具和模型本身：

    - log_before_model（@before_model）：每次调用模型前记录消息条数与最后一条消息，追踪模型输入；
    - monitor_tool（@wrap_tool_call）：包装工具调用，记录工具名/参数/成败；检测到报告信号工具
      fill_context_for_report 后，往 runtime.context 写入 report=True 标志位；
    - report_prompt_switch（@dynamic_prompt）：读取 report 标志位，在报告提示词与主客服提示词之间动态切换。
三者配合完成 ReAct 过程的可观测性，以及"普通对话 ↔ 报告生成"的场景切换。
"""



from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from typing import Callable
from langgraph.types import Command
from utils.logger_handler import logger
from langgraph.runtime import Runtime
from utils.prompts_loader import load_report_prompt,load_system_prompt



@before_model   #发消息给模型前
def log_before_model(
        state: AgentState,
        runtime:Runtime,
):
    """
    调用模型前打日志，便于追踪 ReAct 过程中模型的输入。
    这是 @before_model 钩子：每轮在把消息发给大模型之前自动执行一次。
        :param state: Agent 当前状态，state['messages'] 是累积的消息列表
        :param runtime: 运行时上下文（本函数未使用）
        :return: None，钩子只做日志，不修改状态
    """
    # INFO：记录本次发给模型的消息总条数（对话轮次累积到多少条）
    logger.info(f"[log_before_model]即将调用模型,带有{len(state['messages'])}条消息")

    # DEBUG：记录本次发给模型的最后一条消息内容，方便排查问题
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")

    return None


@wrap_tool_call  # 模型调用工具时
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """
    工具调用监控包装器：在真正执行工具前后打日志，便于追踪 ReAct 过程中调了哪个工具、传了什么参数、成败如何。
        :param request: 工具调用请求对象，内含 tool_call（name 工具名、args 传入参数）
        :param handler: 被包装的真正工具处理函数，调用它才会执行原工具
        :return: 工具执行结果（ToolMessage 工具消息 或 Command 命令）
    """
    logger.info(f"[monitor_tool]执行工具: {request.tool_call['name']}")     # 记录要执行的工具名（ReAct 的"行动"步骤）
    logger.info(f"[monitor_tool]传入参数: {request.tool_call['args']}")     # 记录传入参数（ReAct 的"思考"步骤）

    try:
        # 调用真正的工具处理函数，把请求传给它执行
        result = handler(request)
        logger.info(f"[monitor_tool]工具{request.tool_call['name']}调用成功")

        # 特殊处理：如果刚执行的是"报告信号工具"，就在运行时上下文里打一个报告标志位
        if request.tool_call['name'] == 'fill_context_for_report':
            request.runtime.context['report'] = True    # 往 runtime.context 写入 report=True 标记

        return result   # 原样返回工具结果，交给后续 ReAct 循环
    except Exception as e:
        # 工具执行失败：记录错误日志（含工具名和原因），再把异常重新抛出
        logger.error(f"[monitor_tool]工具{request.tool_call['name']}调用失败,原因: {str(e)}")
        raise e


@dynamic_prompt   # 每一次生成提示词前
def report_prompt_switch(request: ModelRequest):
    """
    动态提示词切换：根据运行时上下文里的 report 标志位，决定给模型哪套提示词。
        :param request: 模型请求对象，可通过 request.runtime.context 读取运行时状态
        :return: 返回要使用的提示词字符串（报告提示词 或 主系统提示词）
    """
    # 读取 report 标志位；.get 第二个参数 False 表示没有这个键时默认 False（非报告场景）
    is_report = request.runtime.context.get("report",False)

    # 标志位为 True（已调用过 fill_context_for_report）→ 切换到报告生成提示词
    if is_report:
        return load_report_prompt()

    # 默认（普通对话场景）→ 使用主客服系统提示词
    return load_system_prompt()