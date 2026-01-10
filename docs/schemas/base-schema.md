# 基础 JSON 结构说明

低代码平台使用 JSON 描述 UI 组件树、属性配置和事件处理。本文档定义了 JSON Schema 的基本结构和规范。

## 基本结构

一个低代码页面的基本 JSON 结构如下：

```json
{
  "version": "1.0.0",
  "componentName": "Page",
  "props": {},
  "css": "",
  "children": [],
  "dataSource": {},
  "state": {},
  "methods": {},
  "lifeCycles": {}
}
```

## 核心概念

### 组件描述

每个组件描述遵循以下结构：

```json
{
  "componentName": "Button",
  "id": "button_1",
  "props": {
    "text": "点击",
    "type": "primary"
  },
  "events": {},
  "children": []
}
```

| 字段名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| componentName | string | 是 | 组件类型名称，必须与组件库中的组件名一致 |
| id | string | 否 | 组件唯一标识，如不提供则自动生成 |
| props | object | 否 | 组件属性配置对象 |
| events | object | 否 | 组件事件处理配置 |
| children | array | 否 | 子组件列表 |
| condition | string/object | 否 | 条件渲染表达式 |
| loop | object | 否 | 循环渲染配置 |

### 页面状态管理

```json
{
  "state": {
    "count": 0,
    "userInfo": {
      "name": "",
      "age": null
    },
    "list": []
  }
}
```

状态可以在页面的任何组件中通过表达式引用，例如 `{{state.count}}`。

### 数据源配置

```json
{
  "dataSource": {
    "userList": {
      "type": "api",
      "options": {
        "url": "/api/users",
        "method": "GET",
        "params": {
          "pageSize": 10
        }
      },
      "autoFetch": true
    }
  }
}
```

| 字段名 | 类型 | 说明 |
|-------|------|------|
| type | string | 数据源类型，可选值：'api', 'static', 'function' |
| options | object | 数据源配置选项 |
| autoFetch | boolean | 是否在页面加载时自动请求数据 |

### 方法定义

```json
{
  "methods": {
    "handleClick": {
      "type": "function",
      "args": ["event"],
      "body": "state.count += 1;"
    }
  }
}
```

## 数据绑定

低代码平台支持在属性中使用表达式绑定数据：

```json
{
  "props": {
    "text": "静态文本",
    "visible": "{{state.showButton}}",
    "className": "{{state.isActive ? 'active' : 'inactive'}}",
    "style": {
      "color": "{{state.buttonColor}}",
      "fontSize": "{{state.fontSize}}px"
    }
  }
}
```

## 数据结构校验

所有 JSON 结构需符合 JSON Schema 规范，用于校验数据结构的有效性。开发者可使用 schema 验证工具确保配置符合规范。

## 版本控制

| 版本号 | 发布日期 | 主要变更 |
|-------|---------|---------|
| 1.0.0 | 2025-10-01 | 初始版本 |
| 1.1.0 | 2025-12-15 | 添加条件渲染和循环渲染支持 |

## 最佳实践

1. 组件 ID 保持唯一性，便于调试和追踪
2. 合理组织页面状态，避免过度复杂的状态结构
3. 将可复用逻辑封装为公共方法
4. 组件嵌套层级不宜过深，一般不超过 5 层
5. 数据绑定表达式保持简洁，复杂逻辑应封装为方法
