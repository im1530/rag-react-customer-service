"""
文档预处理工具模块（RAG 知识库入库前的通用工具集）

本模块提供 RAG 项目中文档入库前所需的基础工具：
    1. get_file_md5：计算文件内容的 MD5 值，用于文件去重与变更检测；
    2. listdir_with_allowed_types：扫描文件夹，按后缀白名单收集符合要求的文档完整路径；
    3. pdf_loader / txt_loader：把 PDF、TXT 文件统一加载成 LangChain Document 对象，供后续切块、向量化、写入向量库使用。
"""


import os
import hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader


# 函数:获取文件内容的MD5的十六进制字符串
def get_file_md5(file_path: str):
    """
    获取文件内容的MD5值
        :param file_path:目标文件的路径（字符串）
        :return:文件内容计算出的32位十六进制MD5字符串；路径不存在或不是文件/读取出错时返回 None
    """
    # 校验1：路径是否存在
    if not os.path.exists(file_path):
        logger.error(f"[md5计算]文件{file_path}不存在")
        return

    # 校验2：判断它是不是一个文件（排除文件夹等）
    if not os.path.isfile(file_path):
        logger.error(f"[md5计算]路径{file_path}不是一个文件")
        return

    # 创建一个空的 MD5 哈希对象，准备逐块喂入数据
    md5_obj = hashlib.md5()

    # 分块处理避免大文件一次性读进内存
    chunk_size = 4096
    try:
        with open(file_path, 'rb') as f:    # 以二进制只读模式打开
            chunk = f.read(chunk_size)      # 先读第一块
            while chunk:                    # 只要读到内容（chunk 非空）就循环
                md5_obj.update(chunk)       # 把这一块数据更新进 md5 对象参与计算
                chunk = f.read(chunk_size)  # 再读下一块，读到文件末尾时返回 b'' 循环结束

            md5_hex = md5_obj.hexdigest()   # 所有块读完后，生成 32 位十六进制的 md5 字符串
            return md5_hex                  # 返回最终 md5 值

    except Exception as e:
        logger.error(f"[md5计算]文件{file_path}计算md5值出错:{e}")


# 函数:获取文件夹内的文件的完整路径(允许的文件类型)
def listdir_with_allowed_types(dirpath:str,allowed_file_types:tuple[str]):
    """
    遍历指定文件夹，挑出后缀在白名单内的文件，返回它们的完整路径（元组）
        :param dirpath:要扫描的文件夹路径
        :param allowed_file_types:允许的文件类型 --- 后缀元组，如 (".pdf", ".txt", ".md")
        :return:符合后缀的文件完整路径组成的元组
    """
    # 收集符合条件的文件列表
    compliant_files = []

    # 校验：dirpath 不是一个文件夹时，记错误日志并退出
    if not os.path.isdir(dirpath):
        logger.error(f"[listdir_with_allowed_type] {dirpath}不是一个文件夹")
        return ()

    # 遍历文件夹下每一项（仅当前这一层，不递归子文件夹）
    for file in os.listdir(dirpath):
        # 文件名后缀命中allowed_file_types里任意一个就保留（endswith 可直接接收元组）
        if file.endswith(allowed_file_types):
            full_filepath = os.path.join(dirpath,file)  # 把文件夹路径和文件名拼成完整路径
            compliant_files.append(full_filepath)       # 加入文件列表

    # 列表转成元组返回（不可变，防止外部误改）
    return tuple(compliant_files)


# 函数:加载 PDF 文档
def pdf_loader(filepath:str,password=None) -> list[Document]:
    """
    加载 PDF 文件，把每一页解析成一个 LangChain Document 对象
        :param filepath: PDF 文件路径
        :param password: 加密 PDF 的打开密码，默认 None（无密码）
        :return: Document 列表，每个元素对应一页，内含正文文本和元数据
    """
    return PyPDFLoader(filepath,password).load()


# 函数:加载 TXT 纯文本文档
def txt_loader(filepath:str) -> list[Document]:
    """
    加载纯文本文件，解析成 LangChain Document 对象
        :param filepath: txt 文件路径
        :return: Document 列表（一般整篇文本对应一个 Document）
    """
    return TextLoader(filepath,encoding='utf-8').load()
