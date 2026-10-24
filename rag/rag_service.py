"""
RAG 检索总结服务模块

封装基于本地知识库的检索增强生成（RAG）流程：
    - 连接 VectorStoreService 拿到向量库检索器，按问题语义相似度召回相关 Document；
    - 读取 RAG 总结提示词并编译成 PromptTemplate；
    - 组装「提示词模板 → 聊天大模型 → StrOutputParser」调用链；
    - rag_summarize(query)：先检索相关文档、拼接成上下文，再填入模板交给模型生成纯文本回答。
"""



from rag.vector_store import VectorStoreService
from utils.prompts_loader import load_rag_prompt
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# 函数：打印提示词
def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagSummarizeService():
    """RAG 检索总结服务：根据用户问题检索知识库相关文档，拼进提示词，交给大模型生成回答。"""

    # 初始化方法：组装好检索器、提示词、模型和链，供后续调用
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()


    # 初始化链式调用：提示词模板 → 大模型 → 输出解析器
    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain


    # 检索：用问题去向量库里召回相关 Document 列表
    def retriever_documents(self,query:str) -> list[Document]:
        return self.retriever.invoke(query)


    # RAG 总结主流程：检索资料 → 拼成上下文 → 交给链生成回答
    def rag_summarize(self,query:str) -> str:
        # 1. 先检索出相关文档
        context_documents = self.retriever_documents(query)

        # 2. 把多篇文档拼成一段带编号的上下文文本（正文 + 元数据）
        context = ""
        count = 0
        for document in context_documents:
            count += 1
            context += f"【参考资料{count}】: 参考资料:{document.page_content} | 参考元数据:{document.metadata}\n"

        # 3. 把用户问题 input 和拼好的 context 填入模板，走链生成最终回答字符串
        return self.chain.invoke(
            {
                "input": query ,
                "context": context
            }
        )



# 测试
if __name__ == "__main__":
    rag_service = RagSummarizeService()
    print(rag_service.rag_summarize("小户型适合哪些扫地机器人"))
