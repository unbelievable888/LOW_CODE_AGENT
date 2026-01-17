import os
import json
import re
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import networkx as nx
from tenacity import retry, stop_after_attempt, wait_fixed
from langchain.schema.runnable import RunnablePassthrough

class KnowledgeGraphRAGProcessor:
    """
    基于知识图谱的RAG处理器
    结合了向量检索和知识图谱的能力，提供更加结构化和关联性强的检索结果
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化知识图谱RAG处理器
        
        Args:
            api_key: OpenAI API密钥，如果为None则从环境变量获取
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API密钥未提供，请设置OPENAI_API_KEY环境变量或在初始化时提供")
        
        # 初始化嵌入模型
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # 初始化LLM
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=self.api_key
        )
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # 初始化向量存储
        self.vector_store = None
        
        # 初始化知识图谱
        self.knowledge_graph = nx.DiGraph()
        
        # 文档存储路径
        self.db_path = os.path.join(os.path.dirname(__file__), "vector_db")
        os.makedirs(self.db_path, exist_ok=True)
        
        # 知识图谱存储路径
        self.kg_path = os.path.join(os.path.dirname(__file__), "knowledge_graph")
        os.makedirs(self.kg_path, exist_ok=True)
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        加载文档
        
        Args:
            file_paths: 文档路径列表
            
        Returns:
            加载的文档列表
        """
        documents = []
        for file_path in file_paths:
            if file_path.endswith('.md'):
                loader = UnstructuredMarkdownLoader(file_path)
                documents.extend(loader.load())
            else:
                # 可以根据需要添加其他类型的文档加载器
                raise ValueError(f"不支持的文件类型: {file_path}")
        
        return documents
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        处理文档，分割成块
        
        Args:
            documents: 文档列表
            
        Returns:
            处理后的文档块列表
        """
        return self.text_splitter.split_documents(documents)
    
    def build_vector_store(self, documents: List[Document]) -> None:
        """
        构建向量存储
        
        Args:
            documents: 文档列表
        """
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        # 保存向量存储
        self.vector_store.save_local(self.db_path)
    
    def load_vector_store(self) -> None:
        """
        加载向量存储
        """
        if os.path.exists(self.db_path):
            # 添加allow_dangerous_deserialization=True参数，允许反序列化pickle文件
            self.vector_store = FAISS.load_local(
                self.db_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            raise FileNotFoundError(f"向量存储不存在: {self.db_path}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def extract_entities_and_relations(self, documents: List[Document]) -> None:
        """
        从文档中提取实体和关系，构建知识图谱
        
        Args:
            documents: 文档列表
        """
        # 使用LLM提取实体和关系
        entity_extraction_template = """
        从以下文本中提取关键实体和它们之间的关系。
        返回JSON格式，包含实体列表和关系列表。
        每个实体应包含id和name。
        每个关系应包含source（源实体id）、target（目标实体id）和relation（关系类型）。
        
        文本:
        {text}
        
        JSON输出:
        """
        
        entity_extraction_prompt = PromptTemplate(
            template=entity_extraction_template,
            input_variables=["text"]
        )
        
        # 使用新的RunnableSequence语法替代LLMChain
        entity_extraction_chain = entity_extraction_prompt | self.llm
        
        # 处理每个文档
        for doc in documents:
            try:
                # 使用invoke方法
                result = entity_extraction_chain.invoke({"text": doc.page_content})
                result_text = result.content if hasattr(result, 'content') else str(result)
                
                # 尝试解析JSON结果，添加更健壮的错误处理
                try:
                    # 查找JSON内容（可能被包裹在其他文本中）
                    json_match = re.search(r'\{[\s\S]*\}', result_text)
                    if json_match:
                        json_str = json_match.group(0)
                        data = json.loads(json_str)
                    else:
                        # 如果没有找到JSON格式内容，创建一个空的数据结构
                        data = {"entities": [], "relations": []}
                        print(f"未找到JSON内容: {result_text[:100]}...")
                    
                    # 添加实体到知识图谱
                    for entity in data.get("entities", []):
                        self.knowledge_graph.add_node(
                            entity["id"],
                            name=entity["name"],
                            document_id=doc.metadata.get("source", "")
                        )
                    
                    # 添加关系到知识图谱
                    for relation in data.get("relations", []):
                        self.knowledge_graph.add_edge(
                            relation["source"],
                            relation["target"],
                            relation=relation["relation"]
                        )
                except json.JSONDecodeError as je:
                    print(f"JSON解析错误: {je}, 原始响应: {result_text[:100]}...")
                except Exception as inner_e:
                    print(f"处理实体和关系时出错: {inner_e}")
            except Exception as e:
                print(f"处理文档时出错: {e}")
        
        # 保存知识图谱
        self.save_knowledge_graph()
    
    def save_knowledge_graph(self) -> None:
        """
        保存知识图谱
        """
        # 将知识图谱转换为可序列化的格式
        graph_data = {
            "nodes": [
                {
                    "id": node_id,
                    "name": data.get("name", ""),
                    "document_id": data.get("document_id", "")
                }
                for node_id, data in self.knowledge_graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "relation": data.get("relation", "")
                }
                for source, target, data in self.knowledge_graph.edges(data=True)
            ]
        }
        
        # 保存为JSON文件
        kg_file = os.path.join(self.kg_path, "knowledge_graph.json")
        with open(kg_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    def load_knowledge_graph(self) -> None:
        """
        加载知识图谱
        """
        kg_file = os.path.join(self.kg_path, "knowledge_graph.json")
        if os.path.exists(kg_file):
            with open(kg_file, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
            
            # 重建知识图谱
            self.knowledge_graph = nx.DiGraph()
            
            # 添加节点
            for node in graph_data["nodes"]:
                self.knowledge_graph.add_node(
                    node["id"],
                    name=node["name"],
                    document_id=node["document_id"]
                )
            
            # 添加边
            for edge in graph_data["edges"]:
                self.knowledge_graph.add_edge(
                    edge["source"],
                    edge["target"],
                    relation=edge["relation"]
                )
        else:
            print(f"知识图谱文件不存在: {kg_file}")
            self.knowledge_graph = nx.DiGraph()
    
    def sanitize_json_string(self, json_str: str) -> str:
        """
        修复常见的JSON格式错误
        
        Args:
            json_str: 可能包含错误的JSON字符串
            
        Returns:
            修复后的JSON字符串
        """
        # 修复单引号
        json_str = json_str.replace("'", '"')
        
        # 修复没有引号的键
        json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', json_str)
        
        # 修复尾部逗号
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        return json_str
    
    def llm_fix_json(self, json_str: str, error_msg: str) -> str:
        """
        使用LLM修复JSON格式错误
        
        Args:
            json_str: 包含错误的JSON字符串
            error_msg: JSON解析错误信息
            
        Returns:
            修复后的JSON字符串
        """
        fix_template = """
        以下是一个包含格式错误的JSON字符串，请修复它并返回正确的JSON格式。
        
        错误信息:
        {error_msg}
        
        JSON字符串:
        {json_str}
        
        修复后的JSON:
        """
        
        fix_prompt = PromptTemplate(
            template=fix_template,
            input_variables=["json_str", "error_msg"]
        )
        
        # 使用新的RunnableSequence语法
        fix_chain = fix_prompt | self.llm
        
        try:
            result = fix_chain.invoke({
                "json_str": json_str,
                "error_msg": error_msg
            })
            result_text = result.content if hasattr(result, 'content') else str(result)
            
            # 查找JSON内容
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                return json_match.group(0)
            else:
                # 如果没有找到JSON格式内容，返回原始字符串
                return json_str
        except Exception as e:
            print(f"使用LLM修复JSON时出错: {e}")
            return json_str
    
    def query_vector_store(self, query: str, k: int = 5) -> List[Document]:
        """
        查询向量存储
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            相关文档列表
        """
        if self.vector_store is None:
            try:
                self.load_vector_store()
            except FileNotFoundError:
                raise ValueError("向量存储未初始化，请先构建向量存储")
        
        # 使用相似度搜索
        docs = self.vector_store.similarity_search(query, k=k)
        return docs
    
    def query_knowledge_graph(self, entity_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        查询知识图谱
        
        Args:
            entity_name: 实体名称
            max_depth: 最大深度
            
        Returns:
            查询结果
        """
        if not self.knowledge_graph.nodes:
            try:
                self.load_knowledge_graph()
            except:
                raise ValueError("知识图谱未初始化，请先构建知识图谱")
        
        # 查找匹配的节点
        matched_nodes = []
        for node_id, data in self.knowledge_graph.nodes(data=True):
            if entity_name.lower() in data.get("name", "").lower():
                matched_nodes.append((node_id, data))
        
        if not matched_nodes:
            return {"entities": [], "relations": []}
        
        # 构建子图
        subgraph_nodes = set()
        for node_id, _ in matched_nodes:
            # BFS遍历
            visited = {node_id}
            queue = [(node_id, 0)]  # (node_id, depth)
            
            while queue:
                current_id, depth = queue.pop(0)
                subgraph_nodes.add(current_id)
                
                if depth < max_depth:
                    # 获取邻居节点
                    neighbors = list(self.knowledge_graph.successors(current_id)) + list(self.knowledge_graph.predecessors(current_id))
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, depth + 1))
        
        # 构建子图
        subgraph = self.knowledge_graph.subgraph(subgraph_nodes)
        
        # 转换为可序列化的格式
        result = {
            "entities": [
                {
                    "id": node_id,
                    "name": data.get("name", ""),
                    "document_id": data.get("document_id", "")
                }
                for node_id, data in subgraph.nodes(data=True)
            ],
            "relations": [
                {
                    "source": source,
                    "target": target,
                    "relation": data.get("relation", "")
                }
                for source, target, data in subgraph.edges(data=True)
            ]
        }
        
        return result
    
    def hybrid_search(self, query: str, k: int = 5, use_kg: bool = True) -> List[Document]:
        """
        混合搜索，结合向量检索和知识图谱
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            use_kg: 是否使用知识图谱增强
            
        Returns:
            相关文档列表
        """
        # 首先进行向量检索
        vector_docs = self.query_vector_store(query, k=k)
        
        if not use_kg:
            return vector_docs
        
        try:
            # 使用LLM提取查询中的实体
            entity_extraction_template = """
            从以下查询中提取关键实体。
            返回JSON格式，包含实体列表。
            每个实体应包含name字段。
            
            查询:
            {query}
            
            JSON输出:
            """
            
            entity_extraction_prompt = PromptTemplate(
                template=entity_extraction_template,
                input_variables=["query"]
            )
            
            # 使用新的RunnableSequence语法
            entity_extraction_chain = entity_extraction_prompt | self.llm
            
            result = entity_extraction_chain.invoke({"query": query})
            result_text = result.content if hasattr(result, 'content') else str(result)
            
            # 解析JSON结果
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                json_str = json_match.group(0)
                try:
                    data = json.loads(json_str)
                    entities = data.get("entities", [])
                except json.JSONDecodeError:
                    # 尝试修复JSON
                    json_str = self.sanitize_json_string(json_str)
                    try:
                        data = json.loads(json_str)
                        entities = data.get("entities", [])
                    except json.JSONDecodeError as je:
                        # 使用LLM修复JSON
                        json_str = self.llm_fix_json(json_str, str(je))
                        try:
                            data = json.loads(json_str)
                            entities = data.get("entities", [])
                        except:
                            entities = []
            else:
                entities = []
            
            # 查询知识图谱
            kg_docs = []
            for entity in entities:
                entity_name = entity.get("name", "")
                if entity_name:
                    kg_result = self.query_knowledge_graph(entity_name, max_depth=1)
                    
                    # 获取相关文档ID
                    doc_ids = set()
                    for node in kg_result["entities"]:
                        doc_id = node.get("document_id", "")
                        if doc_id:
                            doc_ids.add(doc_id)
                    
                    # 根据文档ID查找文档
                    for doc in vector_docs:
                        if doc.metadata.get("source", "") in doc_ids and doc not in kg_docs:
                            kg_docs.append(doc)
            
            # 合并结果，确保知识图谱结果优先
            combined_docs = kg_docs.copy()
            for doc in vector_docs:
                if doc not in combined_docs:
                    combined_docs.append(doc)
            
            return combined_docs[:k]
        except Exception as e:
            print(f"混合搜索时出错: {e}")
            return vector_docs
    
    def process_query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        处理查询，返回结构化的结果
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            包含查询结果的字典
        """
        # 使用混合搜索获取相关文档
        docs = self.hybrid_search(query, k=k, use_kg=True)
        
        # 提取文档内容
        contexts = [doc.page_content for doc in docs]
        
        # 构建提示模板
        qa_template = """
        基于以下上下文回答问题。如果上下文中没有足够的信息来回答问题，请说明无法回答，不要编造答案。
        
        上下文:
        {context}
        
        问题:
        {query}
        
        回答:
        """
        
        qa_prompt = PromptTemplate(
            template=qa_template,
            input_variables=["context", "query"]
        )
        
        # 使用新的RunnableSequence语法
        qa_chain = qa_prompt | self.llm
        
        # 合并上下文
        context = "\n\n".join(contexts)
        
        # 生成回答
        result = qa_chain.invoke({
            "context": context,
            "query": query
        })
        
        answer = result.content if hasattr(result, 'content') else str(result)
        
        # 构建结果
        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        }
    
    def initialize_from_documents(self, file_paths: List[str]) -> None:
        """
        从文档初始化RAG系统
        
        Args:
            file_paths: 文档路径列表
        """
        # 加载文档
        documents = self.load_documents(file_paths)
        
        # 处理文档
        chunks = self.process_documents(documents)
        
        # 构建向量存储
        self.build_vector_store(chunks)
        
        # 提取实体和关系，构建知识图谱
        self.extract_entities_and_relations(chunks)
        
        print(f"初始化完成，处理了 {len(documents)} 个文档，生成了 {len(chunks)} 个文档块")
        print(f"知识图谱包含 {len(self.knowledge_graph.nodes)} 个实体和 {len(self.knowledge_graph.edges)} 个关系")
    
    def setup(self, file_paths: List[str] = None, rebuild: bool = False) -> None:
        """
        设置RAG系统，加载或重建向量库和知识图谱
        
        Args:
            file_paths: 文档路径列表，如果为None则不重建
            rebuild: 是否重建向量库和知识图谱
        """
        try:
            if rebuild and file_paths:
                # 重建向量库和知识图谱
                self.initialize_from_documents(file_paths)
            else:
                # 尝试加载现有向量库和知识图谱
                self.load_vector_store()
                self.load_knowledge_graph()
                print("成功加载现有向量库和知识图谱")
        except Exception as e:
            if file_paths:
                print(f"加载失败，尝试重建: {str(e)}")
                self.initialize_from_documents(file_paths)
            else:
                raise ValueError(f"加载向量库和知识图谱失败，且未提供文档路径进行重建: {str(e)}")
    
    def query(self, query_text: str, return_sources: bool = False) -> Any:
        """
        查询RAG系统
        
        Args:
            query_text: 查询文本
            return_sources: 是否返回源文档
            
        Returns:
            如果return_sources为True，返回(answer, sources)元组
            否则，仅返回answer
        """
        if self.vector_store is None:
            try:
                self.load_vector_store()
            except FileNotFoundError:
                raise ValueError("向量存储未初始化，请先构建向量存储")
            
        print(f"查询: {query_text}")
        
        # 使用混合搜索获取相关文档
        docs = self.hybrid_search(query_text, k=5)
        sources = [doc.page_content for doc in docs]
        
        if return_sources:
            # 构造提示词
            prompt = f"""
            你是一个低代码平台专家助手。请根据提供的上下文回答问题。
            
            上下文信息:
            {' '.join(sources)}
            
            问题: {query_text}
            
            请使用上下文信息回答问题。如果上下文中没有足够信息，请说明你不确定。

            当你返回代码示例时，请特别注意以下几点：
            1. HTML代码：确保所有标签都正确闭合，属性值使用引号包裹
            2. JSON/JavaScript代码：
               a. 确保所有字符串使用双引号包裹
               b. 特别注意处理JavaScript模板字符串（如{{...}}）中的反斜杠
               c. 确保转义序列（如\\n, \\r等）被正确处理
               d. 避免使用可能导致JSON解析错误的控制字符
            
            如果需要提供包含复杂JavaScript模板的代码（如React JSX或低代码配置），请确保它们是有效且正确格式化的。
            """
            
            # 使用LLM生成回答
            result = self.llm.invoke(prompt)
            response = result.content if hasattr(result, 'content') else str(result)
            
            # 检查和处理JSON格式错误
            try:
                # 尝试解析JSON
                json.loads(response)
                print("响应JSON格式正确")
            except json.JSONDecodeError as e:
                print(f"检测到JSON格式错误: {str(e)}")
                # 尝试修复JSON格式
                response = self.sanitize_json_string(response)
                
                # 再次检查
                try:
                    json.loads(response)
                    print("JSON已被成功修复")
                except json.JSONDecodeError as e2:
                    print(f"基本修复失败，尝试高级修复: {str(e2)}")
                    # 使用LLM进行高级修复
                    response = self.llm_fix_json(response, str(e2))
            
            return response, sources
        else:
            # 使用process_query方法处理查询
            result = self.process_query(query_text)
            answer = result["answer"]
            
            # 检查和处理JSON格式错误
            try:
                # 尝试解析JSON
                json.loads(answer)
            except json.JSONDecodeError as e:
                print(f"检测到JSON格式错误: {str(e)}")
                # 尝试修复JSON格式
                answer = self.sanitize_json_string(answer)
                
                # 再次检查
                try:
                    json.loads(answer)
                    print("JSON已被成功修复")
                except json.JSONDecodeError as e2:
                    print(f"基本修复失败，尝试高级修复: {str(e2)}")
                    # 使用LLM进行高级修复
                    answer = self.llm_fix_json(answer, str(e2))
            
            return answer
