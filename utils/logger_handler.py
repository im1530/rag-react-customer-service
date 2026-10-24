"""
项目的日志处理模块,用于处理日志的记录
"""

import os
import logging
from utils.path_tool import get_abs_path
from datetime import datetime


# 日志保存的根目录 --- 日志都存放到这个目录下
LOG_ROOT_PATH = get_abs_path("logs")


# 确保日志的目录存在,不存在则创建
os.makedirs(LOG_ROOT_PATH,exist_ok=True)


# 日志的格式配置
DEFAULT_LOG_FORMAT = logging.Formatter(
    "[%(asctime)s] - 日志器:%(name)s - %(levelname)s - %(filename)s:第%(lineno)d行 - %(message)s"
)
"""
DEFAULT_LOG_FORMAT 日志格式说明（各占位符含义）：
- %(asctime)s    日志产生时间，默认格式 年-月-日 时:分:秒,毫秒
- %(name)s       日志器名称，即 logging.getLogger("名字") 传入的名字
- %(levelname)s  日志级别：DEBUG / INFO / WARNING / ERROR / CRITICAL
- %(filename)s   输出日志的源文件名（不含路径）
- %(lineno)d     输出日志的代码所在行号（d 表示十进制整数）
- %(message)s    日志正文，即 logger.info("这里的内容")
"""


# 函数: 获取日志器
def get_logger(
        name: str = "agent.yml",
        console_level:int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file = None
) -> logging.Logger:
    """
    创建并返回一个能输出日志到 控制台 和 文件 的日志器。
    :param name:日志器名称，对应格式串里的 %(name)s，默认 "agent.yml"
    :param console_level:控制台输出的日志的最低级别，默认 INFO（只显示 INFO 及以上）
    :param file_level:文件里写入的日志的最低级别，默认 DEBUG（记录更详细，DEBUG 及以上）
    :param log_file:日志文件路径，默认 None
    :return:日志器
    """
    logger = logging.getLogger(name)    # 按 name 获取日志器；同名时返回已存在的，避免重复创建
    logger.setLevel(logging.DEBUG)      # 设置日志器的总最低级别为 DEBUG（最高门槛，先放行全部，再由各处理器自行过滤）

    # 如果该日志器已经挂载过处理器（说明重复调用了本函数）,直接返回，避免重复添加 handler 导致同一条日志被打印多次
    if logger.handlers:
        return logger

    # 控制台handler的配置
    console_handler = logging.StreamHandler()   # 创建控制台处理器，用于把日志输出到终端
    console_handler.setLevel(console_level)             # 设置该处理器的输出级别，即函数参数传入的 console_level
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)    # 设置该处理器的日志格式为 DEFAULT_LOG_FORMAT

    # 自动生成一个默认文件路径
    # 如果调用时没有手动传入 log_file 路径,就把日志根目录 LOG_ROOT_PATH 和文件名拼成完整路径来使用
    if not log_file:
        log_file = os.path.join(LOG_ROOT_PATH, f"{name}_{datetime.now().strftime('%Y_%m_%d')}.log")

    # 文件handler的配置
    file_handler = logging.FileHandler(log_file,encoding="utf-8")   # 创建文件处理器，用于把日志输出到文件
    file_handler.setLevel(file_level)                # 设置文件写入级别，即参数 file_level（默认 DEBUG，记录更详细）
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)    # 设置该处理器的日志格式为 DEFAULT_LOG_FORMAT

    # 添加处理器到日志器
    logger.addHandler(console_handler)  # 添加控制台处理器
    logger.addHandler(file_handler)     # 添加文件处理器

    return logger


# 快捷获取日志器,便于在其他模块中使用
logger = get_logger()



# 测试
if __name__ == "__main__":
    logger.error("错误❌️-日志")
    logger.warning("警告⚠️-日志")
    logger.info("信息✅-日志")
    logger.debug("调试🐛-日志")
