import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 
import json
import sys
import time
import logging
import socket
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import requests
from requests.exceptions import RequestException, ConnectionError, Timeout
import httpx
from httpx import ReadTimeout, ConnectTimeout, ConnectError
import re
from bs4 import BeautifulSoup
import html5lib

# 这些导入需要安装相应的包:
# pip install langchain langchain-community faiss-cpu openai sentence-transformers

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.schema.document import Document
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


@dataclass
class RAGConfig:
    """RAG配置类"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # 嵌入模型
    chunk_size: int = 1000  # 文档分块大小
    chunk_overlap: int = 200  # 分块重叠大小
    docs_path: str = "../../docs/"  # 文档路径，指向新的docs目录
    vector_db_path: str = "./vector_db"  # 向量数据库存储路径
    llm_model: str = "claude-3-7-sonnet" # 大语言模型名称
    top_k: int = 5  # 检索的文档数量
    timeout: int = 120  # API请求超时时间（秒）

class LowCodeRAGProcessor:
    """低代码平台RAG处理器"""
    
    def __init__(self, config: RAGConfig = None):
        """初始化RAG处理器
        
        Args:
            config: RAG配置
        """
        self.config = config or RAGConfig()
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.vectorstore = None
        self.llm = None
        
    def load_documents(self) -> List[Document]:
        """加载文档
        
        Returns:
            处理后的文档列表
        """
        print(f"正在从 {self.config.docs_path} 加载文档...")
        
        loaders = []
        
        # 加载Markdown文档
        md_loader = DirectoryLoader(
            self.config.docs_path,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader
        )
        loaders.append(md_loader)
        
        # 加载JSON文档
        class JSONLoader(TextLoader):
            """自定义JSON加载器，提取并美化JSON内容"""
            def load(self) -> List[Document]:
                docs = super().load()
                for doc in docs:
                    try:
                        # 解析和美化JSON
                        parsed = json.loads(doc.page_content)
                        doc.page_content = json.dumps(parsed, indent=2, ensure_ascii=False)
                    except json.JSONDecodeError:
                        pass
                return docs
        
        json_loader = DirectoryLoader(
            self.config.docs_path,
            glob="**/*.json",
            loader_cls=JSONLoader
        )
        loaders.append(json_loader)
        
        # 合并所有文档
        documents = []
        for loader in loaders:
            documents.extend(loader.load())
            
        print(f"加载了 {len(documents)} 个文档")
        
        # 文档分块
        split_documents = self.text_splitter.split_documents(documents)
        print(f"分块后共有 {len(split_documents)} 个文档块")
        
        return split_documents
    
    def build_vectorstore(self, documents: List[Document] = None) -> None:
        """构建向量存储
        
        Args:
            documents: 文档列表，如果为None则加载文档
        """
        if documents is None:
            documents = self.load_documents()
            
        print("正在构建向量存储...")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # 保存向量数据库
        os.makedirs(os.path.dirname(self.config.vector_db_path), exist_ok=True)
        self.vectorstore.save_local(self.config.vector_db_path)
        print(f"向量存储已保存到 {self.config.vector_db_path}")
    
    def load_vectorstore(self) -> bool:
        """加载向量存储
        
        Returns:
            是否成功加载
        """
        if os.path.exists(self.config.vector_db_path):
            print(f"正在加载向量存储 {self.config.vector_db_path}...")
            self.vectorstore = FAISS.load_local(
                self.config.vector_db_path,
                self.embeddings
            )
            return True
        else:
            print("向量存储不存在，请先构建向量存储")
            return False

    def initialize_llm(self, openai_api_key: Optional[str] = None) -> None:
        """初始化大语言模型
        
        Args:
            openai_api_key: OpenAI API密钥，如果为None则使用环境变量
        """
        try:
            # 我们仍然需要保留langchain的LLM接口用于检索QA
            # 检查API密钥
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
            elif "OPENAI_API_KEY" not in os.environ:
                raise ValueError("未提供OpenAI API密钥，请通过参数传入或设置OPENAI_API_KEY环境变量")
            
            # 设置API基础URL和其他参数
            kwargs = {
                "model_name": self.config.llm_model,
                "temperature": 0,
                "request_timeout": self.config.timeout,
                # "base_url": 'https://oneapi.qunhequnhe.com/v1',
                # "api_key": 'sk-HEOZViCV4rzTmbCy66F61b04Fc1d431e84Bd3a1d38Cf24A7'
            }
            
            # 设置httpx选项，传递额外的参数
            os.environ["OPENAI_HTTPX_EXTRA_KWARGS"] = json.dumps({
                "timeout": self.config.timeout
            })
            
            self.llm = ChatOpenAI(**kwargs)
        except Exception as e:
            raise
    
    def setup(self, rebuild_vectorstore: bool = False, openai_api_key: Optional[str] = None) -> None:
        """设置RAG系统
        
        Args:
            rebuild_vectorstore: 是否重新构建向量存储
            openai_api_key: OpenAI API密钥
        """
        if rebuild_vectorstore or not self.load_vectorstore():
            self.build_vectorstore()
        
        self.initialize_llm(openai_api_key)
    
    def validate_html(self, html_content: str) -> str:
        """验证并修复HTML内容中不闭合的DOM元素（兼容性方法）
        
        Args:
            html_content: 原始HTML内容
            
        Returns:
            修复后的HTML内容
        """
        # 检测HTML问题
        issues = self.detect_html_issues(html_content)
        
        # 如果检测到问题，使用LLM修复
        if issues:
            return self.llm_fix_html(html_content, issues)
        else:
            return html_content
    
    def detect_html_issues(self, html_content: str) -> List[str]:
        """检测HTML内容中的问题
        
        Args:
            html_content: 原始HTML内容
            
        Returns:
            问题描述列表
        """
        issues = []
        
        try:
            # 使用html5lib解析HTML，它会尝试修复错误
            original_parser = html5lib.HTMLParser(strict=False)
            original_dom = original_parser.parse(html_content)
            
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(html_content, 'html5lib')
            fixed_html = str(soup)
            
            # 比较修复前后的内容
            if fixed_html != html_content:
                # 检查常见的HTML问题
                
                # 1. 检查未闭合标签
                unclosed_pattern = r'<([a-zA-Z0-9]+)[^>]*?>(?!.*?</\1>)'
                unclosed_tags = re.findall(unclosed_pattern, html_content)
                for tag in set(unclosed_tags):
                    # 排除自闭合标签
                    if tag not in ['img', 'input', 'br', 'hr', 'meta', 'link']:
                        issues.append(f"未闭合的<{tag}>标签")
                
                # 2. 检查未正确嵌套的标签
                nested_pattern = r'<([a-zA-Z0-9]+)[^>]*?>.*?<([a-zA-Z0-9]+)[^>]*?>.*?</\1>.*?</\2>'
                if re.search(nested_pattern, html_content):
                    issues.append("存在交叉嵌套的标签")
                
                # 3. 检查缺少引号的属性
                attr_pattern = r'<[a-zA-Z0-9]+\s+[^>]*?=[^\'"][^\'"\s>]+[^>]*?>'
                if re.search(attr_pattern, html_content):
                    issues.append("属性值缺少引号")
                    
                # 如果没有具体检测到问题但内容有变化，添加一个通用问题描述
                if not issues:
                    issues.append("HTML结构存在问题，可能包括未闭合标签或不正确的嵌套")
        except Exception as e:
            issues.append(f"HTML解析错误: {str(e)}")
            
        return issues
    
    def sanitize_json_string(self, content: str, is_js_template: bool = False) -> str:
        """清理字符串中可能导致JSON解析错误的字符
        
        Args:
            content: 原始字符串
            is_js_template: 是否为JavaScript模板字符串（含有类似{{...}}的模板）
            
        Returns:
            清理后的字符串
        """
        # 替换掉所有JSON中非法的控制字符
        # 这些是ASCII范围0-31的字符，除了常见的\b, \f, \n, \r, \t
        illegal_chars = [chr(i) for i in range(32) if i not in [9, 10, 13]]
        
        for char in illegal_chars:
            content = content.replace(char, '')
            
        # 对于含有JavaScript模板的JSON，我们需要一个完全不同的处理方法
        if is_js_template:
            # 策略：将JSON解析为结构，然后为每个字段单独处理，特别关注含有模板的字段
            try:
                # 尝试直接解析，如果成功就不需要特殊处理
                json.loads(content)
                return content
            except json.JSONDecodeError:
                # 如果解析失败，需要特殊处理
                pass
            
            # 使用更复杂的模式来匹配和保护所有可能的模板内容
            # 这个模式匹配{{...}}以及其中可能包含的所有内容，包括嵌套大括号
            def protect_templates(text):
                # 使用栈来匹配模板
                result = []
                i = 0
                length = len(text)
                
                while i < length:
                    # 查找可能的模板开始
                    template_start = text.find("{{", i)
                    
                    if template_start == -1:
                        # 没有找到更多模板，添加剩余文本
                        result.append(text[i:])
                        break
                    
                    # 添加模板前的文本
                    result.append(text[i:template_start])
                    
                    # 查找匹配的模板结束
                    brace_count = 2  # 已经找到了两个左大括号
                    j = template_start + 2
                    
                    while j < length and brace_count > 0:
                        if text[j:j+2] == "{{":
                            brace_count += 2
                            j += 2
                        elif text[j:j+2] == "}}":
                            brace_count -= 2
                            j += 2
                        else:
                            j += 1
                    
                    if brace_count == 0:
                        # 找到了匹配的模板结束
                        template = text[template_start:j]
                        
                        # 保护模板中的特殊字符
                        protected_template = template
                        # 先保护反斜杠
                        protected_template = protected_template.replace("\\", "###BACKSLASH###")
                        # 保护反引号中的模板字符串 ${...}
                        protected_template = re.sub(r'`(.*?)`', lambda m: m.group(0).replace('${', '###TEMPLATE_START###').replace('}', '###TEMPLATE_END###'), protected_template)
                        # 保护JSX标签
                        protected_template = re.sub(r'<([A-Z][a-zA-Z0-9]*)', r'###JSX_TAG_START###\1', protected_template)
                        protected_template = re.sub(r'</([A-Z][a-zA-Z0-9]*)', r'###JSX_TAG_END###\1', protected_template)
                        protected_template = protected_template.replace('>', '###JSX_TAG_CLOSE###')
                        
                        result.append(protected_template)
                        i = j
                    else:
                        # 没有找到匹配的模板结束，将剩余文本作为普通文本处理
                        result.append(text[template_start:])
                        break
                
                return "".join(result)
            
            # 保护模板
            protected_content = protect_templates(content)
            
            # 处理非模板部分
            # 1. 确保引号使用是一致的
            protected_content = re.sub(r'(?<!\\)"', '\\"', protected_content)
            protected_content = protected_content.replace('\\"', '"')
            
            # 2. 处理非模板部分的反斜杠
            # 找到所有的###BACKSLASH###并替换为一个临时标记以避免被下面的替换影响
            protected_content = protected_content.replace("###BACKSLASH###", "###TEMP_BACKSLASH_MARK###")
            
            # 现在替换所有反斜杠为双反斜杠
            protected_content = protected_content.replace("\\", "\\\\")
            
            # 修复可能的过度转义
            protected_content = protected_content.replace('\\\\"', '\\"')
            
            # 恢复被保护的所有特殊标记
            final_content = protected_content.replace("###TEMP_BACKSLASH_MARK###", "\\")
            # 恢复模板字符串中的${...}
            final_content = final_content.replace("###TEMPLATE_START###", "${").replace("###TEMPLATE_END###", "}")
            # 恢复JSX标签
            final_content = final_content.replace("###JSX_TAG_START###", "<").replace("###JSX_TAG_END###", "</").replace("###JSX_TAG_CLOSE###", ">")
            
            # 处理可能的转义序列
            # 确保转义序列在JSON字符串中被正确处理
            for seq in ["\\n", "\\r", "\\t", "\\b", "\\f"]:
                # 只替换那些不在模板中的转义序列
                parts = final_content.split("{{")
                processed_parts = [parts[0].replace(seq, "\\" + seq)]
                
                for i in range(1, len(parts)):
                    if "}}" in parts[i]:
                        template_part, rest = parts[i].split("}}", 1)
                        # 模板部分保持不变，非模板部分替换转义序列
                        processed_parts.append("{{" + template_part + "}}" + rest.replace(seq, "\\" + seq))
                    else:
                        # 如果没有关闭的模板，保持不变
                        processed_parts.append("{{" + parts[i])
                
                final_content = "".join(processed_parts)
            
            return final_content
        else:
            # 正常处理非JavaScript模板字符串
            # 确保引号使用是一致的
            content = re.sub(r'(?<!\\)"', '\\"', content)
            content = content.replace('\\"', '"')
            
            # 处理反斜杠
            content = content.replace('\\', '\\\\')
            content = content.replace('\\\\"', '\\"')
            
            # 修复常见的控制字符转义
            content = content.replace('\\n', '\\\\n')
            content = content.replace('\\r', '\\\\r')
            content = content.replace('\\t', '\\\\t')
            
            return content
    
    def llm_fix_json(self, json_content: str, error_message: str) -> str:
        """使用LLM修复JSON解析错误
        
        Args:
            json_content: 原始JSON内容
            error_message: JSON解析错误信息
            
        Returns:
            修复后的JSON内容
        """
        # 检查是否包含React/JSX语法（<Tag>形式）
        has_jsx = bool(re.search(r'<[A-Z][a-zA-Z]*', json_content))
        
        # 构建提示词，请求LLM修复JSON
        # 构建提示词，请求LLM修复JSON
        jsx_example = r'return text === "active" ? \<Tag color=\"green\"\>上架\</Tag\> : \<Tag color=\"red\"\>下架\</Tag\>;'
        
        prompt = f"""
        你是一个JSON/JavaScript专家。以下代码存在JSON解析错误，请修复这些错误并返回完整的修复后代码。
        不要添加任何说明或解释，只返回修复后的代码。
        
        错误信息：
        {error_message}
        
        需要修复的代码：
        ```json
        {json_content}
        ```
        
        {"检测到代码中包含React/JSX语法（如<Tag>）。" if has_jsx else ""}
        
        修复时请特别注意：
        1. 修复控制字符问题，如处理不当的换行符、制表符等
        2. 确保所有字符串使用双引号包裹
        3. 确保JavaScript模板字符串（如{{...}}）中的反斜杠被正确处理
        4. 确保所有反斜杠转义符号被正确处理
        5. 检查并修复可能导致错误的特殊字符
        6. {"对于包含React/JSX语法的代码（如<Tag>），确保它们在JSON中被正确表示为字符串并转义" if has_jsx else ""}
        7. {"在JSON中，JSX应该被处理为字符串，所有<和>符号前要加反斜杠，例如：" + jsx_example if has_jsx else ""}
        8. {"确保转义后的字符串中不包含未经处理的控制字符" if has_jsx else ""}
        
        只返回修复后的代码，不要有任何额外解释：
        """
        
        try:
            # 使用LLM修复JSON
            response = self.llm.invoke(prompt).content
            
            # 提取修复后的代码
            code_pattern = r"```(?:json|javascript)?\s*([\s\S]*?)```"
            code_match = re.search(code_pattern, response)
            if code_match:
                fixed_json = code_match.group(1).strip()
            else:
                fixed_json = response.strip()
                
            # 验证修复后的JSON是否可以解析
            try:
                json.loads(fixed_json)
                print("LLM成功修复JSON，验证通过")
            except json.JSONDecodeError as e:
                print(f"LLM修复后的JSON仍然存在问题: {str(e)}")
                # 尝试更强力的修复
                return self.llm_fix_json_advanced(fixed_json, str(e))
                
            return fixed_json
        except Exception as e:
            print(f"LLM修复JSON过程出错: {str(e)}")
            # 出错时返回原始内容
            return json_content
    
    def llm_fix_json_advanced(self, json_content: str, error_message: str) -> str:
        """高级JSON修复尝试，用于处理普通修复失败的情况
        
        Args:
            json_content: 已经尝试修复但仍有问题的JSON内容
            error_message: JSON解析错误信息
            
        Returns:
            修复后的JSON内容
        """
        # 检查是否包含React/JSX语法（<Tag>形式）
        has_jsx = bool(re.search(r'<[A-Z][a-zA-Z]*', json_content))
        
        # 更详细的提示词，要求LLM逐步分析和修复问题
        # 更详细的提示词，要求LLM逐步分析和修复问题
        jsx_example = r'return text === "active" ? \<Tag color=\"green\"\>上架\</Tag\> : \<Tag color=\"red\"\>下架\</Tag\>;'
        
        prompt = f"""
        你是一位JSON修复专家。之前的修复尝试失败了，现在需要你进行深入分析和修复。
        
        错误信息：
        {error_message}
        
        有问题的JSON：
        ```json
        {json_content}
        ```
        
        {"我检测到JSON中包含React/JSX语法（如<Tag>元素）。这是一个常见的问题来源，因为JSX不是有效的JSON语法。" if has_jsx else ""}
        
        请执行以下步骤修复：
        1. 仔细分析错误位置和类型
        2. 特别关注控制字符、转义序列和JavaScript模板字符串
        3. 确保所有字符串正确使用双引号，并且所有引号都正确闭合
        4. 处理所有特殊字符和转义序列
        5. 确保所有括号、大括号和中括号都正确配对
        6. {"对于包含React/JSX的代码（如<Tag>元素），必须将所有<和>符号转义为反斜杠+<和反斜杠+>，或将整个JSX表达式转换为字符串表示" if has_jsx else ""}
        7. {"例如：" + jsx_example if has_jsx else ""}
        8. {"或者考虑将JSX表达式转换为更符合JSON语法的表示形式，如使用数组或对象表示组件层次结构" if has_jsx else ""}
        9. 确保修复后的代码能被JSON.parse()函数成功解析
        
        只返回最终修复后的JSON代码，不要有任何解释：
        """
        
        try:
            # 使用LLM进行高级修复
            response = self.llm.invoke(prompt).content
            
            # 提取修复后的代码
            code_pattern = r"```(?:json|javascript)?\s*([\s\S]*?)```"
            code_match = re.search(code_pattern, response)
            if code_match:
                fixed_json = code_match.group(1).strip()
            else:
                fixed_json = response.strip()
                
            print("LLM完成高级JSON修复")
            return fixed_json
        except Exception as e:
            print(f"LLM高级修复JSON过程出错: {str(e)}")
            # 出错时返回原始内容
            return json_content
    
    def llm_fix_html(self, html_content: str, issues: List[str]) -> str:
        """使用LLM修复HTML内容
        
        Args:
            html_content: 原始HTML内容
            issues: 检测到的问题列表
            
        Returns:
            修复后的HTML内容
        """
        if not issues:
            return html_content
            
        # 构建提示词，请求LLM修复HTML
        issues_text = "\n".join([f"- {issue}" for issue in issues])
        
        prompt = f"""
        你是一个HTML专家。以下HTML代码存在一些问题，请修复这些问题并返回完整的修复后代码。
        不要添加任何说明或解释，只返回修复后的HTML代码。
        重要提示：
        1. 不要在HTML内容中包含任何控制字符（如原始的换行符、制表符等）
        2. 确保所有的属性值都使用引号包裹
        3. 确保所有的JSON内容都是有效的，不含有非法字符
        4. 所有引号应该配对使用，不应有未闭合的引号
        
        检测到的问题:
        {issues_text}
        
        原始HTML代码:
        ```
        {html_content}
        ```
        
        修复后的HTML代码:
        """
        
        try:
            # 使用LLM修复HTML
            response = self.llm.invoke(prompt).content
            
            # 提取修复后的代码
            code_pattern = r"```(?:html|HTML)?\s*([\s\S]*?)```"
            code_match = re.search(code_pattern, response)
            if code_match:
                fixed_html = code_match.group(1).strip()
            else:
                fixed_html = response.strip()
            
            # 清理HTML内容中可能导致JSON解析错误的字符
            fixed_html = self.sanitize_json_string(fixed_html)
                
            print("LLM已修复HTML问题")
            return fixed_html
        except Exception as e:
            print(f"LLM修复HTML过程出错: {str(e)}")
            # 出错时返回原始内容
            return html_content
    
    def fix_jsx_in_json(self, json_content: str) -> str:
        """专门处理JSON中的JSX/React语法
        
        Args:
            json_content: 包含JSX的JSON内容
            
        Returns:
            修复后的JSON内容
        """
        # 构建专门针对JSX的提示词
        before_example = r'"body": "return text === \"active\" ? <Tag color=\"green\">上架</Tag> : <Tag color=\"red\">下架</Tag>;"'
        after_example = r'"body": "return text === \"active\" ? \<Tag color=\"green\"\>上架\</Tag\> : \<Tag color=\"red\"\>下架\</Tag\>;"'
        
        prompt = f"""
        你是一个JSON与React/JSX专家。以下JSON代码中包含React JSX语法，导致无法正确解析。
        请将所有JSX元素（如<Tag>）转换为有效的JSON字符串表示。
        
        需要修复的代码：
        ```json
        {json_content}
        ```
        
        修复指南：
        1. 识别所有的JSX标签（通常以<大写字母>开始）
        2. 将JSX表达式中的所有<和>符号转义为反斜杠+<和反斜杠+>
        3. 确保所有引号正确嵌套和转义
        4. 修复所有控制字符问题
        
        举例：
        原始: {before_example}
        修复后: {after_example}
        
        只返回修复后的JSON代码，不要有任何解释：
        """
        
        try:
            # 使用LLM修复JSX
            response = self.llm.invoke(prompt).content
            
            # 提取修复后的代码
            code_pattern = r"```(?:json)?\s*([\s\S]*?)```"
            code_match = re.search(code_pattern, response)
            if code_match:
                fixed_json = code_match.group(1).strip()
            else:
                fixed_json = response.strip()
            
            # 验证修复后的代码
            try:
                json.loads(fixed_json)
                print("JSX修复成功，JSON验证通过")
                return fixed_json
            except json.JSONDecodeError as e:
                print(f"JSX修复后仍有问题: {str(e)}")
                return json_content
        except Exception as e:
            print(f"修复JSX过程出错: {str(e)}")
            return json_content
    
    def post_process_response(self, response: str) -> str:
        """对LLM生成的响应进行后处理
        
        Args:
            response: LLM原始响应
            
        Returns:
            处理后的响应
        """
        # 识别响应中的代码块
        code_pattern = r"```(?:html|HTML|json|JSON|javascript|JAVASCRIPT)?\s*([\s\S]*?)```"
        
        def replace_code_block(match):
            # 获取代码块内容
            code_content = match.group(1).strip()
            code_type = match.group(0).split('`')[3].strip().lower() if len(match.group(0).split('`')) > 3 else "html"
            
            # 清理所有代码块中的特殊字符
            cleaned_content = self.sanitize_json_string(code_content)
            
            # 如果是HTML代码块，检测并修复HTML问题
            if code_type == "html" or code_type == "":
                issues = self.detect_html_issues(cleaned_content)
                
                # 如果检测到问题，使用LLM修复
                if issues:
                    print(f"检测到HTML问题: {issues}")
                    fixed_content = self.llm_fix_html(cleaned_content, issues)
                    return f"```html\n{fixed_content}\n```"
                else:
                    return f"```html\n{cleaned_content}\n```"
            
            # 如果是JSON或JavaScript代码块，使用LLM修复可能的解析问题
            elif "json" in code_type or "javascript" in code_type:
                try:
                    # 首先尝试解析
                    parsed_json = json.loads(code_content)
                    # 如果成功，返回美化的JSON
                    formatted_json = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                    return f"```{code_type}\n{formatted_json}\n```"
                except json.JSONDecodeError as e:
                    # 检查是否包含JSX语法
                    has_jsx = bool(re.search(r'<[A-Z][a-zA-Z]*', code_content))
                    
                    if has_jsx:
                        # 如果包含JSX，使用专门的JSX修复方法
                        print(f"检测到JSX在JSON中，尝试修复: {str(e)}")
                        fixed_content = self.fix_jsx_in_json(code_content)
                    else:
                        # 否则使用一般的JSON修复方法
                        print(f"JSON解析错误: {str(e)}")
                        fixed_content = self.llm_fix_json(code_content, str(e))
                    
                    # 再次验证修复后的内容
                    try:
                        json.loads(fixed_content)
                        print("JSON修复成功，验证通过")
                    except json.JSONDecodeError as e2:
                        print(f"修复后仍有问题，尝试高级修复: {str(e2)}")
                        fixed_content = self.llm_fix_json_advanced(fixed_content, str(e2))
                        
                    return f"```{code_type}\n{fixed_content}\n```"
            else:
                # 其他类型的代码块，直接返回清理后的内容
                return f"```{code_type}\n{cleaned_content}\n```"
        
        # 替换所有代码块
        processed_response = re.sub(code_pattern, replace_code_block, response)
        
        # 检查响应中是否包含未包装在代码块中的JSON字符串
        # 这对于直接在响应文本中包含的JSON很有用
        json_pattern = r'(\{[\s\S]*?\})'
        
        def sanitize_potential_json(match):
            potential_json = match.group(1)
            try:
                # 尝试解析为JSON
                json.loads(potential_json)
                # 如果成功解析，说明这是一个有效的JSON，进行清理
                return self.sanitize_json_string(potential_json)
            except json.JSONDecodeError:
                # 如果解析失败，不是有效的JSON，保持原样
                return potential_json
        
        # 尝试查找和清理响应中可能的JSON对象
        processed_response = re.sub(json_pattern, sanitize_potential_json, processed_response)
        
        return processed_response
            
    def query(self, query: str, return_sources: bool = False) -> Any:
        """查询RAG系统
        
        Args:
            query: 查询文本
            return_sources: 是否返回源文档
            
        Returns:
            查询结果或(查询结果, 源文档)
        """
        if self.vectorstore is None:
            raise ValueError("向量存储未初始化，请先调用setup方法")
            
        print(f"查询: {query}")
        
        # 创建检索器
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.top_k}
        )
        
        # 检索文档
        if return_sources:
            docs = retriever.get_relevant_documents(query)
            sources = [doc.page_content for doc in docs]
            
            # 构造提示词
            prompt = f"""
            你是一个低代码平台专家助手。请根据提供的上下文回答问题。
            
            上下文信息:
            {' '.join(sources)}
            
            问题: {query}
            
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
            
            # 使用langchain的llm实例进行查询
            response = self.llm.invoke(prompt).content
            
            # 对响应进行HTML校验和修复
            processed_response = self.post_process_response(response)
            
            return processed_response, sources
        else:
            # 使用RetrievalQA链
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=retriever,
                return_source_documents=False
            )
            
            response = qa_chain.invoke({"query": query})
            result = response["result"]
            
            # 对响应进行HTML校验和修复
            processed_result = self.post_process_response(result)
            
            return processed_result
    
    def semantic_search(self, query: str, top_k: int = None) -> List[Document]:
        """语义搜索
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            相关文档列表
        """
        if self.vectorstore is None:
            raise ValueError("向量存储未初始化，请先调用setup方法")
            
        if top_k is None:
            top_k = self.config.top_k
            
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": top_k}
        )
        
        return retriever.get_relevant_documents(query)


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
    rag = LowCodeRAGProcessor(config)
    
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
