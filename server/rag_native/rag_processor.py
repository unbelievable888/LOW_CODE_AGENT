"""
原生RAG处理器 - 不依赖LangChain的实现
"""

import os
import json
import sys
import time
import logging
import re
import glob
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import numpy as np

# 外部依赖
import faiss
import openai
import requests
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

# 文档处理
import markdown
from unstructured.partition.text import partition_text
from unstructured.partition.md import partition_md


@dataclass
class RAGConfig:
    """RAG配置类"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # 嵌入模型
    chunk_size: int = 1000  # 文档分块大小
    chunk_overlap: int = 200  # 分块重叠大小
    docs_path: str = "../../docs/"  # 文档路径
    vector_db_path: str = "./vector_db"  # 向量数据库存储路径
    llm_model: str = "gpt-3.5-turbo"  # 大语言模型名称
    top_k: int = 5  # 检索的文档数量
    timeout: int = 120  # API请求超时时间（秒）
    api_base: str = 'https://oneapi.qunhequnhe.com/v1'  # API基础URL
    api_key: str = 'sk-HEOZViCV4rzTmbCy66F61b04Fc1d431e84Bd3a1d38Cf24A7'  # API密钥


class Document:
    """文档类，表示一个文本块及其元数据"""
    
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        """
        初始化文档
        
        Args:
            page_content: 文档内容
            metadata: 文档元数据
        """
        self.page_content = page_content
        self.metadata = metadata or {}


class TextSplitter:
    """文本分割器，将文本分割成多个块"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        初始化文本分割器
        
        Args:
            chunk_size: 块大小
            chunk_overlap: 块重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """
        分割文本
        
        Args:
            text: 要分割的文本
            
        Returns:
            分割后的文本块列表
        """
        # 使用简单的字符计数来分割文本
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            # 找到适当的分割点
            end = min(start + self.chunk_size, len(text))
            
            # 如果不是在文本末尾，尝试在句子或段落边界处分割
            if end < len(text):
                # 尝试在段落边界处分割
                paragraph_end = text.rfind('\n\n', start, end)
                if paragraph_end > start + self.chunk_size // 2:
                    end = paragraph_end + 2
                else:
                    # 尝试在句子边界处分割
                    sentence_end = text.rfind('. ', start, end)
                    if sentence_end > start + self.chunk_size // 2:
                        end = sentence_end + 2
            
            # 添加块
            chunks.append(text[start:end])
            
            # 更新起点，考虑重叠
            start = end - self.chunk_overlap
            
            # 确保起点有效
            if start < 0:
                start = 0
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档
        
        Args:
            documents: 要分割的文档列表
            
        Returns:
            分割后的文档列表
        """
        split_docs = []
        
        for doc in documents:
            splits = self.split_text(doc.page_content)
            
            for i, split in enumerate(splits):
                # 创建新元数据，添加分块信息
                metadata = doc.metadata.copy()
                metadata["chunk"] = i
                
                # 创建新文档
                split_docs.append(Document(split, metadata))
        
        return split_docs


class DocumentLoader:
    """文档加载器，加载各种格式的文档"""
    
    @staticmethod
    def load_markdown(file_path: str) -> Document:
        """
        加载Markdown文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档对象
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 解析Markdown为纯文本
            html = markdown.markdown(content)
            text = re.sub('<[^<]+?>', '', html)  # 简单去除HTML标签
            
            return Document(text, {"source": file_path, "type": "markdown"})
        except Exception as e:
            print(f"加载Markdown文件 {file_path} 失败: {str(e)}")
            return Document("", {"source": file_path, "error": str(e)})
    
    @staticmethod
    def load_json(file_path: str) -> Document:
        """
        加载JSON文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档对象
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 解析并美化JSON
            parsed = json.loads(content)
            pretty_json = json.dumps(parsed, indent=2, ensure_ascii=False)
            
            return Document(pretty_json, {"source": file_path, "type": "json"})
        except Exception as e:
            print(f"加载JSON文件 {file_path} 失败: {str(e)}")
            return Document("", {"source": file_path, "error": str(e)})
    
    @staticmethod
    def load_directory(directory_path: str) -> List[Document]:
        """
        加载目录中的所有文档
        
        Args:
            directory_path: 目录路径
            
        Returns:
            文档对象列表
        """
        documents = []
        
        # 加载Markdown文件
        md_files = glob.glob(os.path.join(directory_path, "**/*.md"), recursive=True)
        for file_path in md_files:
            documents.append(DocumentLoader.load_markdown(file_path))
        
        # 加载JSON文件
        json_files = glob.glob(os.path.join(directory_path, "**/*.json"), recursive=True)
        for file_path in json_files:
            documents.append(DocumentLoader.load_json(file_path))
        
        return documents


class Embeddings:
    """嵌入处理类，用于将文本转换为向量"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        初始化嵌入处理器
        
        Args:
            model_name: 模型名称
        """
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        嵌入文档
        
        Args:
            texts: 要嵌入的文本列表
            
        Returns:
            文本嵌入向量列表
        """
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        嵌入查询
        
        Args:
            text: 要嵌入的文本
            
        Returns:
            文本嵌入向量
        """
        return self.model.encode(text).tolist()


class FAISSVectorStore:
    """FAISS向量存储，用于存储和检索文档向量"""
    
    def __init__(self, embedding_size: int = 384):
        """
        初始化FAISS向量存储
        
        Args:
            embedding_size: 嵌入向量维度
        """
        self.index = faiss.IndexFlatL2(embedding_size)  # 使用L2距离度量
        self.documents = []
        self.embedding_size = embedding_size
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """
        添加文档
        
        Args:
            documents: 要添加的文档列表
            embeddings: 文档嵌入向量列表
        """
        if len(documents) != len(embeddings):
            raise ValueError(f"文档数量 ({len(documents)}) 与嵌入数量 ({len(embeddings)}) 不匹配")
            
        # 存储文档
        start_idx = len(self.documents)
        self.documents.extend(documents)
        
        # 添加向量
        vectors = np.array(embeddings).astype('float32')
        self.index.add(vectors)
    
    def similarity_search(self, query_vector: List[float], k: int = 5) -> List[Tuple[Document, float]]:
        """
        相似度搜索
        
        Args:
            query_vector: 查询向量
            k: 返回的文档数量
            
        Returns:
            相似文档及其距离的列表
        """
        vector = np.array([query_vector]).astype('float32')
        
        # 执行搜索
        distances, indices = self.index.search(vector, k)
        
        # 返回文档及其距离
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:
                results.append((self.documents[idx], distances[0][i]))
        
        return results
    
    def save(self, directory: str) -> None:
        """
        保存向量存储
        
        Args:
            directory: 保存目录
        """
        os.makedirs(directory, exist_ok=True)
        
        # 保存FAISS索引
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        
        # 保存文档
        with open(os.path.join(directory, "index.pkl"), 'wb') as f:
            pickle.dump(self.documents, f)
    
    @classmethod
    def load(cls, directory: str, embedding_size: int = 384) -> 'FAISSVectorStore':
        """
        加载向量存储
        
        Args:
            directory: 加载目录
            embedding_size: 嵌入向量维度
            
        Returns:
            加载的向量存储
        """
        vector_store = cls(embedding_size)
        
        # 加载FAISS索引
        vector_store.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        
        # 加载文档
        with open(os.path.join(directory, "index.pkl"), 'rb') as f:
            vector_store.documents = pickle.load(f)
            
        return vector_store


class LLMClient:
    """LLM客户端，处理与OpenAI API的通信"""
    
    def __init__(self, 
                 model: str = "gpt-3.5-turbo", 
                 api_key: str = None,
                 api_base: str = None,
                 timeout: int = 120):
        """
        初始化LLM客户端
        
        Args:
            model: 模型名称
            api_key: API密钥
            api_base: API基础URL
            timeout: 超时时间（秒）
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout
        
        # 设置OpenAI客户端
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            
        Returns:
            回复内容
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                timeout=self.timeout
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM请求失败: {str(e)}")
            raise


class NativeRAGProcessor:
    """原生RAG处理器，不依赖LangChain"""
    
    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG处理器
        
        Args:
            config: RAG配置
        """
        self.config = config or RAGConfig()
        self.embeddings = Embeddings(model_name=self.config.embedding_model)
        self.text_splitter = TextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.vector_store = None
        self.llm_client = None
    
    def load_documents(self) -> List[Document]:
        """加载文档
        
        Returns:
            处理后的文档列表
        """
        print(f"正在从 {self.config.docs_path} 加载文档...")
        
        # 加载文档
        documents = DocumentLoader.load_directory(self.config.docs_path)
        
        # 过滤空文档
        documents = [doc for doc in documents if doc.page_content.strip()]
        
        print(f"加载了 {len(documents)} 个文档")
        
        # 文档分块
        split_documents = self.text_splitter.split_documents(documents)
        print(f"分块后共有 {len(split_documents)} 个文档块")
        
        return split_documents
    
    def build_vector_store(self, documents: List[Document] = None) -> None:
        """构建向量存储
        
        Args:
            documents: 文档列表，如果为None则加载文档
        """
        if documents is None:
            documents = self.load_documents()
            
        print("正在构建向量存储...")
        
        # 提取文本内容
        texts = [doc.page_content for doc in documents]
        
        # 计算嵌入
        embeddings = self.embeddings.embed_documents(texts)
        
        # 创建向量存储
        embedding_size = len(embeddings[0])
        self.vector_store = FAISSVectorStore(embedding_size=embedding_size)
        
        # 添加文档
        self.vector_store.add_documents(documents, embeddings)
        
        # 保存向量数据库
        os.makedirs(os.path.dirname(self.config.vector_db_path), exist_ok=True)
        self.vector_store.save(self.config.vector_db_path)
        print(f"向量存储已保存到 {self.config.vector_db_path}")
    
    def load_vector_store(self) -> bool:
        """加载向量存储
        
        Returns:
            是否成功加载
        """
        if os.path.exists(os.path.join(self.config.vector_db_path, "index.faiss")):
            print(f"正在加载向量存储 {self.config.vector_db_path}...")
            
            # 加载嵌入维度
            sample_text = "Sample text for embedding size detection"
            embedding_size = len(self.embeddings.embed_query(sample_text))
            
            # 加载向量存储
            self.vector_store = FAISSVectorStore.load(
                self.config.vector_db_path,
                embedding_size=embedding_size
            )
            
            return True
        else:
            print("向量存储不存在，请先构建向量存储")
            return False
    
    def initialize_llm(self) -> None:
        """初始化LLM客户端"""
        self.llm_client = LLMClient(
            model=self.config.llm_model,
            api_key=self.config.api_key,
            api_base=self.config.api_base,
            timeout=self.config.timeout
        )
    
    def setup(self, rebuild_vector_store: bool = False) -> None:
        """设置RAG系统
        
        Args:
            rebuild_vector_store: 是否重新构建向量存储
        """
        if rebuild_vector_store or not self.load_vector_store():
            self.build_vector_store()
        
        self.initialize_llm()
    
    def query(self, query: str, return_sources: bool = False) -> Any:
        """查询RAG系统
        
        Args:
            query: 查询文本
            return_sources: 是否返回源文档
            
        Returns:
            查询结果或(查询结果, 源文档)
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化，请先调用setup方法")
            
        if self.llm_client is None:
            raise ValueError("LLM客户端未初始化，请先调用setup方法")
            
        print(f"查询: {query}")
        
        # 嵌入查询
        query_embedding = self.embeddings.embed_query(query)
        
        # 检索文档
        search_results = self.vector_store.similarity_search(
            query_embedding, 
            k=self.config.top_k
        )
        
        # 提取文档内容
        documents = [doc for doc, _ in search_results]
        sources = [doc.page_content for doc in documents]
        
        # 构造提示词
        system_message = "你是一个低代码平台专家助手。请根据提供的上下文回答问题。如果上下文中没有足够信息，请说明你不确定。"
        
        context = "\n\n".join(sources)
        user_message = f"""
        上下文信息:
        {context}
        
        问题: {query}
        
        请使用上下文信息回答问题。如果上下文中没有足够信息，请说明你不确定。
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # 发送请求
        response = self.llm_client.chat_completion(messages)
        
        if return_sources:
            return response, sources
        else:
            return response


def run_demo():
    """运行演示"""
    # 配置
    config = RAGConfig(
        docs_path="../../docs/",  # 文档路径
        vector_db_path="./vector_db_native",  # 向量数据库存储路径
        api_base='https://oneapi.qunhequnhe.com/v1',
        api_key='sk-HEOZViCV4rzTmbCy66F61b04Fc1d431e84Bd3a1d38Cf24A7',
        chunk_size=500,  # 减小块大小，降低内存使用
        chunk_overlap=50  # 减小重叠大小
    )
    
    # 初始化RAG处理器
    rag = NativeRAGProcessor(config)
    
    # 设置RAG系统
    try:
        # 检查是否已经存在向量数据库，如果存在则直接加载
        if os.path.exists(os.path.join(config.vector_db_path, "index.faiss")):
            print("检测到现有向量数据库，直接加载...")
            rag.setup(rebuild_vector_store=False)
        else:
            rag.setup(rebuild_vector_store=True)
    except Exception as e:
        print(f"设置RAG系统失败: {str(e)}")
        sys.exit(1)
    
    try:
        query = "搜索列表页面怎么实现？"
        response, sources = rag.query(query, return_sources=True)
            
        print("\n" + "="*50)
        print(f"查询: {query}")
        print("-"*50)
        print(f"回答: {response}")
        print("-"*50)
        print("参考来源:")
        for i, source in enumerate(sources[:3]):  # 只显示前3个来源
            print(f"来源 {i+1}:")
            print(source[:200] + "..." if len(source) > 200 else source)
            print()
    except Exception as e:
        print(f"查询 '{query}' 失败: {str(e)}")


if __name__ == "__main__":
    run_demo()
