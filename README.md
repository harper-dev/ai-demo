# AI Demo Project

一个用于学习和实践 AI 相关技术的演示项目，主要基于 LangChain 和 LangGraph 框架。

## 📚 项目概述

本项目包含多个 AI 应用示例，涵盖了从基础的语言模型使用到复杂的多步骤 AI 代理系统。每个文件都专注于特定的 AI 概念和技术实现。

## 🚀 快速开始

### 环境要求
- Python 3.11+
- OpenAI API Key 或其他支持的模型 API

### 安装依赖
```bash
pip install -r requirements.txt
```

### 环境配置
创建 `.env` 文件并添加必要的 API 密钥：
```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## 📁 项目结构

### 核心示例文件

| 文件名 | 功能描述 | 学习重点 |
|--------|----------|----------|
| `model_demo.py` | 基础语言模型使用 | 模型初始化、工具绑定、批处理 |
| `weather_agent.py` | 简单天气查询代理 | Agent 创建、工具调用 |
| `defining_tools.py` | 工具定义和错误处理 | 自定义工具、中间件 |
| `customer_support.py` | 多步骤客服系统 | 状态管理、流程控制、Command |
| `semantic_search.py` | 语义搜索系统 | 文档加载、向量化、相似性搜索 |
| `personal_agents.py` | 个人助理代理 | 日历管理、邮件发送 |
| `rag_agent.py` | RAG 检索增强生成 | 文档检索、知识库问答 |
| `sql_agent.py` | SQL 查询代理 | 数据库交互、自然语言转SQL |
| `deep_agent.py` | 深度学习代理 | 复杂推理、多轮对话 |

### 配置文件
- `requirements.txt` - Python 依赖包
- `.env` - 环境变量配置
- `langgraph.json` - LangGraph 配置

### 文档资源
- `doc/` - 包含学习资料和测试文档
  - `nke-10k-2023.pdf` - Nike 财报（用于语义搜索测试）
  - `Agentic_AI_Engineer_Learning_Roadmap.pdf` - AI 工程师学习路线图

## 🎯 学习路径

### 1. 基础概念 (开始这里)
```bash
python model_demo.py          # 了解基础模型使用
python weather_agent.py       # 学习简单 Agent 创建
python defining_tools.py      # 掌握工具定义
```

### 2. 中级应用
```bash
python semantic_search.py     # 语义搜索和向量数据库
python personal_agents.py     # 多功能个人助理
python sql_agent.py          # 数据库交互
```

### 3. 高级系统
```bash
python customer_support.py    # 复杂状态管理
python rag_agent.py          # 检索增强生成
python deep_agent.py         # 深度推理系统
```

## 🔧 核心技术栈

- **LangChain**: 语言模型应用开发框架
- **LangGraph**: 多步骤 AI 工作流构建
- **OpenAI/Anthropic**: 大语言模型 API
- **Ollama**: 本地模型部署
- **Vector Stores**: 向量数据库和语义搜索
- **Tools & Agents**: 工具调用和智能代理

## 💡 关键学习点

### 1. Agent 系统设计
- 状态管理 (`AgentState`)
- 工具定义和调用 (`@tool`)
- 中间件使用 (`@wrap_model_call`)
- 流程控制 (`Command`)

### 2. 语义搜索
- 文档加载和分割
- 向量化和嵌入
- 相似性搜索
- 检索增强生成 (RAG)

### 3. 多步骤工作流
- 条件路由
- 状态传递
- 错误处理
- 检查点和持久化

## 🛠️ 运行示例

### 客服系统演示
```bash
python customer_support.py
```
体验多步骤客服流程：保修验证 → 问题分类 → 解决方案

### 语义搜索演示
```bash
python semantic_search.py
```
在 Nike 财报中搜索相关信息

### 天气查询代理
```bash
python weather_agent.py
```
简单的天气查询对话

## 📖 学习资源

- [LangChain 官方文档](https://docs.langchain.com/)
- [LangGraph 教程](https://docs.langchain.com/oss/python/langgraph/)
- [OpenAI API 文档](https://platform.openai.com/docs)
