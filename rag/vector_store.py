"""
向量库服务模块（VectorStoreService）

封装 Chroma 向量数据库的初始化与知识库的增量构建，是 RAG 检索与入库的核心服务：
    - __init__：连接 Chroma 向量库（集合名、持久化目录、嵌入模型），并初始化文本切分器；
    - get_retriever：基于向量库生成检索器，按相似度(k条)召回相关文档，供问答链使用；
    - load_document：增量加载知识库——扫描 data 目录，按白名单后缀收集文档，
                    通过内容 MD5 登记簿去重，仅对新增/改动文件做加载、切块、向量化并写入向量库，
                    入库成功后登记 MD5，实现不重复、可增量更新。

依赖：chroma_conf 配置、embedding_model 嵌入模型、文件加载器/MD5/目录扫描工具。
"""


import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.file_handler import txt_loader,pdf_loader,listdir_with_allowed_types
from utils.file_handler import get_file_md5


class VectorStoreService:
    # 初始化 Chroma 向量库连接
    def __init__(self):
        self.vector_store = Chroma(
            collection_name = chroma_conf["collection_name"],       # 集合名（相当于一张表），向量存放在该集合
            persist_directory = get_abs_path(chroma_conf["persist_directory"]),   # 向量落盘保存目录，关闭程序后仍保留
            embedding_function = embedding_model,                  # 嵌入模型
        )

        # 初始化文本分割器
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],               # 每块最大字符数
            chunk_overlap=chroma_conf["chunk_overlap"],         # 相邻块重叠字符数
            separators=chroma_conf["separators"],               # 切分优先级分隔符列表，优先在这些位置断开
            length_function = len,                              # 计算长度的函数
        )


    # 方法:创建向量检索器
    def get_retriever(self):
        """
        返回向量检索器,会根据用户问题检索向量数据库,获取相似度最高的文档 ---- .as_retriever: 可以返回一个Runnable接口的子类实例对象,能够入链
        """
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["similarity_k"]})


    #
    def load_document(self):
        """
            加载知识库：扫描数据目录，对新文件做去重、加载、切块并向量化入库（增量更新）。

            整体流程：
                1. 扫描 data 目录，按白名单后缀（txt/pdf）收集所有文档完整路径；
                2. 逐个文件计算内容 MD5，与登记簿(md5_hex_store)比对去重：已登记的文件跳过，不重复入库；
                3. 对未处理文件按后缀加载成 Document（txt/pdf），读取失败则跳过；
                4. 用文本切分器切块，切块为空则跳过；
                5. add_documents 将小块向量化并写入 Chroma 向量库；
                6. 入库成功后把该文件 MD5 追加登记，下次启动只处理新增/改动文件；
                7. 单个文件出错仅记错误日志，不中断其它文件处理。
            :return: None
        """
        # 方法：检查某个文件的 MD5 是否已经处理过(去重)
        def check_md5_hex(md5_for_check:str):
            """
            检查某个文件的 MD5 是否已经处理过
            :param md5_for_check: 要进行检查的 MD5 值
            :return: 布尔值，表示该 MD5 值是否已经处理过
            """
            # 情况1：MD5 记录文件本身不存在，说明是首次运行，先创建一个空记录文件
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]),"w",encoding="utf-8").close()
                return False    # 记录为空，该 md5 没处理过

            # 情况2：MD5 记录文件存在，读取文件内容并检查
            with open(get_abs_path(chroma_conf["md5_hex_store"]),"r",encoding="utf-8") as f:
                for line in f.readlines():         # 逐行取出已登记的 md5
                    line = line.strip()            # 去掉每行首尾的换行符/空格
                    if line == md5_for_check:      # 命中相同的 md5
                        return True                # 该 md5 已处理过
                return False                # 遍历完都没匹配上，说明没处理过


        # 方法：将某个 MD5 值添加到记录文件中
        def add_md5_hex(md5_to_add:str):
            """
            将某个 MD5 值添加到记录文件中
            :param md5_to_add: 要添加的 MD5 值
            :return: 无
            """
            with open(get_abs_path(chroma_conf["md5_hex_store"]),"a",encoding="utf-8") as f:
                f.write(md5_to_add + "\n")


        # 方法:把某个文件转换成 LangChain Document 列表
        def turn_file_to_document(filepath_for_turn:str):
            """
            将某个文件的内容转换为list[Document]
            :param filepath_for_turn: 要转换的文件路径
            :return: list[Document],读取不到返回空列表
            """
            # 后缀是 txt → 交给纯文本加载器
            if filepath_for_turn.endswith('txt'):
                return txt_loader(filepath_for_turn)

            # 后缀是 pdf → 交给 PDF 加载器（每页一个 Document）
            if filepath_for_turn.endswith('pdf'):
                return pdf_loader(filepath_for_turn)

            # 其他后缀 → 无法处理，返回空列表
            return []


        # 扫描数据文件夹，拿到白名单后缀的文档的完整路径
        types_allowed_file_path: list[str] = listdir_with_allowed_types(
            get_abs_path(chroma_conf["data_path"]),                           # 第一个参数：要扫描的数据目录（如 data）
            tuple(chroma_conf["allowed_knowledge_file_types"])  # 第二个参数：允许的文件后缀，转成元组传给函数
        )

        # 遍历每个符合类型的文件路径，逐个做去重、加载、切块、入库
        for filepath in types_allowed_file_path:
            # 获取文件的md5值
            md5_hex = get_file_md5(filepath)

            # 已处理过（登记簿里已有该 md5）→ 记日志并跳过本轮，不重复入库
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{filepath} 内容已存在知识库中，跳过")
                continue

            try:
                # 按文件后缀加载成 Document 列表（txt 走 txt_loader，pdf 走 pdf_loader）
                documents: list[Document] = turn_file_to_document(filepath)
                # 加载结果为空（读不到/不支持的类型）→ 警告并跳过
                if not documents:
                    logger.warning(f"[加载知识库]{filepath} 无法读取，跳过")
                    continue

                # 用切分器把长文档切成小块，得到 split_document
                split_document: list[Document] = self.spliter.split_documents(documents)
                # 切块结果为空（没有有效内容）→ 警告并跳过
                if not split_document:
                    logger.warning(f"[加载知识库]{filepath} 分片后没有有效内容，跳过")
                    continue

                # 添加向量数据到向量数据库中：内部自动用嵌入模型转向量并写入 Chroma
                self.vector_store.add_documents(split_document)

                # 入库成功后，把该文件的 md5 追加登记到登记簿，下次启动就不会重复处理
                add_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{filepath} 内容添加到向量数据库成功")

            except Exception as e:
                # 整个加载/入库过程出错：记录错误日志，exc_info=True 额外打印完整堆栈便于排查
                logger.error(f"[加载知识库]{filepath} 添加向量数据库时出错：{str(e)}", exc_info=True)




# 测试
if __name__ == "__main__":
    vss = VectorStoreService()
    vss.load_document()
    retriever = vss.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print('-'*20)

