"""
模型工厂模块（工厂模式）

用统一的抽象接口 create_model() 封装聊天模型与嵌入模型的创建过程：
    - BaseModelFactory：抽象基类，规定所有模型工厂都必须实现 create_model()；
    - ChatModelFactory：创建通义千问聊天模型（ChatTongyi），用于对话与 RAG 总结；
    - EmbeddingModelFactory：创建通义嵌入模型（DashScopeEmbeddings），用于把文本转向量。
模型名均从 rag_conf 配置读取，后续换厂商/模型只需改对应工厂类，调用处不变。
"""


from abc import ABC, abstractmethod
from typing import Optional
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from utils.config_handler import rag_conf


# 抽象类:模型工厂的统一规范
class BaseModelFactory(ABC):
    """所有模型工厂的抽象基类，约定必须实现 create_model()"""
    @abstractmethod
    def create_model(self) -> Optional[Embeddings | BaseChatModel]:
        """
        创建并返回一个模型实例
            :return: 嵌入模型(Embeddings) 或 聊天模型(BaseChatModel)，也可能为 None
        """
        pass  # 抽象方法，基类不实现，交给子类各自覆盖


# 具体类:嵌入模型工厂
class EmbeddingModelFactory(BaseModelFactory):
    """创建文本嵌入模型，负责把文本转换成向量"""
    def create_model(self) -> Optional[Embeddings]:
        return DashScopeEmbeddings(model=rag_conf["text_embedding_model_name"])


# 具体类:聊天模型工厂
class ChatModelFactory(BaseModelFactory):
    """创建聊天大模型，负责对话与生成回答"""
    def create_model(self) -> Optional[BaseChatModel]:
        return ChatTongyi(model=rag_conf["chat_model_name"])


# 模块加载时直接造出两个模型实例，便于供其它模块导入使用
chat_model = ChatModelFactory().create_model()              # 聊天大模型
embedding_model = EmbeddingModelFactory().create_model()    # 文本嵌入模型



