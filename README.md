# 低代码平台 (LOW_CODE_AGENT)

## 项目介绍

低代码平台是一个智能化开发工具，旨在通过RAG (检索增强生成) 技术提供智能组件推荐、代码生成和文档检索功能，帮助开发者更高效地构建应用程序。系统利用向量数据库存储和检索文档信息，通过大语言模型生成上下文相关的高质量回答。

## 目录结构

```
LOW_CODE_AGENT/
├── docs/                     # 文档和知识库
│   ├── components/           # 组件文档
│   ├── examples/             # 示例文档
│   └── schemas/              # 模式定义
├── server/                   # 服务端代码
│   ├── rag_langchain/        # 基于LangChain实现的RAG系统
│   │   ├── rag_processor.py  # RAG核心处理逻辑
│   │   ├── requirements.txt  # 依赖项列表
│   │   └── vector_db/        # 向量数据库存储目录
│   ├── rag_native/           # 原生实现的RAG系统
│   │   ├── rag_processor.py  # 原生RAG处理逻辑
│   │   ├── requirements.txt  # 依赖项列表
│   │   └── vector_db/        # 向量数据库存储目录
│   ├── templates/            # Web服务的前端模板
│   │   └── index.html        # RAG查询的Web界面
│   ├── config.template.py    # API配置模板
│   ├── rag_web_service.py    # RAG Web服务入口
│   └── requirements.txt      # 主要依赖项列表
└── .gitignore                # Git忽略文件配置
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourname/LOW_CODE_AGENT.git
cd LOW_CODE_AGENT
```

### 2. 安装依赖

安装所有依赖（包含Web服务和RAG所需的全部依赖）：

```bash
cd server
pip3 install -r requirements.txt
```

### 3. 配置API密钥

RAG系统需要配置OpenAI API密钥才能正常工作。有两种方式可以提供API密钥：

1. **使用配置文件** (推荐):

   - 复制配置模板文件并修改:
   ```bash
   cd server
   cp config.template.py config.py
   ```
   
   - 编辑`config.py`文件，填入您的API密钥:
   ```python
   OPENAI_API_KEY = "your-api-key-here"  # 替换为你的真实API密钥
   OPENAI_API_BASE = "https://oneapi.qunhequnhe.com/v1"  # 根据需要修改API基础URL
   ```

2. **使用环境变量**:

   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   export OPENAI_API_BASE="https://oneapi.qunhequnhe.com/v1"  # 可选
   ```

**注意**: 切勿将包含真实API密钥的配置文件提交到版本控制系统中。建议将`config.py`添加到`.gitignore`文件中。

### 4. 运行RAG演示（命令行方式）

#### LangChain版本

```bash
cd server
python3 rag_langchain/rag_processor.py
```

#### 原生版本（性能较高但内存要求更高）

```bash
cd server
python3 rag_native/rag_processor.py
```


### 5. 运行RAG Web服务

1. **启动Web服务**:

```bash
cd server
python3 rag_web_service.py
```

2. **访问Web界面**:

打开浏览器，访问 http://localhost:5000

3. **API用法**:

Web服务提供以下API端点：

- **健康检查**: `GET /api/health`
- **执行查询**: `POST /api/query`
  ```json
  {
    "query": "搜索列表页面怎么实现？",
    "include_sources": true
  }
  ```

响应格式：
```json
{
  "success": true,
  "response": "回答内容...",
  "sources": ["来源1...", "来源2..."]
}
```

## 向量数据库

系统在首次运行时会自动初始化向量数据库：

1. **初始化过程**: 系统会自动加载`docs`目录下的文档，进行切分和向量化，存储到向量数据库中
2. **存储位置**: 向量数据库文件存储在`server/rag_langchain/vector_db/`或`server/rag_native/vector_db/`目录下
3. **重建向量库**: 如果文档更新，可以通过设置`rebuild_vectorstore=True`参数重建向量库

**注意**: 请确保不要提交包含敏感信息的文件，如`config.py`。
