"""
RAG Web服务 - 将RAG功能作为API端点提供
"""

import os
import sys
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rag_langchain.rag_processor import LowCodeRAGProcessor, RAGConfig

# 尝试导入配置文件
try:
    from config import OPENAI_API_KEY, OPENAI_API_BASE, DEBUG
    print("已从config.py加载配置")
except ImportError:
    print("未找到config.py文件，将尝试从环境变量中获取配置")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://oneapi.qunhequnhe.com/v1")
    DEBUG = os.environ.get("DEBUG", "True").lower() in ["true", "1", "t", "yes"]

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 启用CORS支持，允许前端访问

# 全局RAG处理器实例
rag_processor = None

def init_rag_processor():
    """初始化RAG处理器"""
    global rag_processor
    
    if rag_processor is not None:
        return
    
    try:
        # 检查API密钥是否已配置
        if not OPENAI_API_KEY:
            raise ValueError(
                "未找到API密钥。请通过以下方式之一设置API密钥:\n"
                "1. 创建config.py文件并设置OPENAI_API_KEY\n"
                "2. 设置环境变量OPENAI_API_KEY\n"
                "参考config.template.py文件获取配置示例。"
            )
        
        # 设置环境变量
        os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        
        # 使用脚本的绝对路径来确定docs和vector_db路径
        script_dir = Path(__file__).resolve().parent
        docs_path = script_dir.parent / "docs"
        vector_db_path = script_dir / "rag_langchain" / "vector_db"
        
        print(f"文档路径: {docs_path}")
        print(f"向量数据库路径: {vector_db_path}")
        
        # 检查文档路径是否存在
        if not docs_path.exists():
            raise FileNotFoundError(f"文档目录不存在: {docs_path}")
        
        # 配置
        config = RAGConfig(
            docs_path=str(docs_path),  # 使用绝对路径
            vector_db_path=str(vector_db_path)  # 使用绝对路径
        )
        
        # 初始化处理器
        rag_processor = LowCodeRAGProcessor(config)
        
        # 设置（尝试加载现有向量库，如果不存在则创建）
        try:
            rag_processor.setup(rebuild_vectorstore=False, openai_api_key=OPENAI_API_KEY)
            print("成功加载现有向量库")
        except Exception as e:
            print(f"加载向量库失败，尝试重建: {str(e)}")
            rag_processor.setup(rebuild_vectorstore=True, openai_api_key=OPENAI_API_KEY)
            print("成功重建向量库")
    
    except Exception as e:
        print(f"初始化RAG处理器失败: {str(e)}")
        raise


@app.route('/')
def home():
    """首页路由，提供简单的UI"""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    """查询API端点"""
    try:
        # 确保RAG处理器已初始化
        global rag_processor
        if rag_processor is None:
            init_rag_processor()
        
        # 获取请求数据
        data = request.json
        query_text = data.get('query', '')
        include_sources = data.get('include_sources', True)
        
        if not query_text:
            return jsonify({
                'success': False,
                'error': '查询文本不能为空'
            }), 400
        
        # 执行查询
        if include_sources:
            response, sources = rag_processor.query(query_text, return_sources=True)
            
            # 检查和处理JSON格式错误
            try:
                # 尝试解析JSON
                json.loads(response)
                print("响应JSON格式正确")
            except json.JSONDecodeError as e:
                print(f"检测到JSON格式错误: {str(e)}")
                # 尝试修复JSON格式
                response = rag_processor.sanitize_json_string(response)
                
                # 再次检查
                try:
                    json.loads(response)
                    print("JSON已被成功修复")
                except json.JSONDecodeError as e2:
                    print(f"基本修复失败，尝试高级修复: {str(e2)}")
                    # 使用LLM进行高级修复
                    response = rag_processor.llm_fix_json(response, str(e2))
            
            # 截断来源以减少响应大小
            truncated_sources = [s[:500] + '...' if len(s) > 500 else s for s in sources]
            return jsonify({
                'success': True,
                'response': response,
                'sources': truncated_sources
            })
        else:
            response = rag_processor.query(query_text, return_sources=False)
            
            # 检查和处理JSON格式错误
            try:
                # 尝试解析JSON
                json.loads(response)
            except json.JSONDecodeError as e:
                print(f"检测到JSON格式错误: {str(e)}")
                # 尝试修复JSON格式
                response = rag_processor.sanitize_json_string(response)
                
                # 再次检查
                try:
                    json.loads(response)
                    print("JSON已被成功修复")
                except json.JSONDecodeError as e2:
                    print(f"基本修复失败，尝试高级修复: {str(e2)}")
                    # 使用LLM进行高级修复
                    response = rag_processor.llm_fix_json(response, str(e2))
            
            return jsonify({
                'success': True,
                'response': response
            })
    
    except Exception as e:
        print(f"查询处理出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == "__main__":
    # 初始化RAG处理器
    print("正在初始化RAG处理器...")
    try:
        init_rag_processor()
        print("RAG处理器初始化成功")
    except Exception as e:
        print(f"RAG处理器初始化失败: {str(e)}")
    
    # 启动Flask应用
    app.run(debug=DEBUG, host='0.0.0.0')
