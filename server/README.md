# 低代码平台 服务端

本目录包含低代码平台Agent服务端实现

## 目录结构

- `rag_langchain/`: 基于LangChain实现的RAG (检索增强生成) 系统
  - `rag_processor.py`: RAG核心处理逻辑
  - `requirements.txt`: LangChain版本的依赖项列表
- `rag_native/`: 原生实现的RAG系统（不依赖LangChain）
  - `rag_processor.py`: 原生RAG处理逻辑
  - `requirements.txt`: 原生版本的依赖项列表
- `templates/`: Web服务的前端模板
  - `index.html`: RAG查询的Web界面
- `rag_web_service.py`: RAG Web服务入口
- `requirements.txt`: 主要依赖项列表（包含Web服务和RAG所需的所有依赖）

## 运行方式

### 安装依赖

安装所有依赖（包含Web服务和RAG所需的全部依赖）：

```bash
cd server
pip3 install -r requirements.txt
```

### 运行RAG演示（命令行方式）

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

### 配置API密钥

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

### 运行RAG Web服务


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

