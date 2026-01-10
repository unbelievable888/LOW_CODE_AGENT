# 事件响应规范

低代码平台提供了强大的事件系统，用于处理用户交互和组件间通信。本文档定义了事件处理的规范和最佳实践。

## 事件处理基本结构

事件处理配置在组件的 `events` 对象中定义：

```json
{
  "componentName": "Button",
  "props": {
    "text": "提交"
  },
  "events": {
    "onClick": {
      "type": "function",
      "value": "handleSubmit"
    }
  }
}
```

## 事件类型

低代码平台支持多种类型的事件处理机制：

### 函数引用型

引用在页面 methods 中定义的函数：

```json
{
  "onClick": {
    "type": "function",
    "value": "handleClick"
  }
}
```

### 内联代码型

直接在事件处理器中编写代码：

```json
{
  "onChange": {
    "type": "jsExpression",
    "value": "function(value) { state.formData.name = value; }"
  }
}
```

### 动作序列型

执行一系列预定义动作：

```json
{
  "onClick": {
    "type": "actionFlow",
    "value": [
      {
        "type": "setState",
        "params": {
          "path": "loading",
          "value": true
        }
      },
      {
        "type": "executeDataSource",
        "params": {
          "name": "submitForm"
        }
      },
      {
        "type": "setState",
        "params": {
          "path": "loading",
          "value": false
        }
      },
      {
        "type": "showMessage",
        "params": {
          "type": "success",
          "content": "提交成功"
        }
      }
    ]
  }
}
```

## 支持的动作类型

| 动作类型 | 说明 | 参数 |
|---------|------|------|
| setState | 修改页面状态 | path: 状态路径, value: 新值 |
| executeDataSource | 执行数据源请求 | name: 数据源名称, params: 请求参数(可选) |
| showMessage | 显示消息提示 | type: 消息类型, content: 消息内容 |
| navigate | 页面导航 | url: 目标地址, target: 打开方式 |
| openModal | 打开模态框 | name: 模态框名称, params: 传递参数 |
| closeModal | 关闭模态框 | name: 模态框名称 |
| confirmDialog | 显示确认对话框 | title: 标题, content: 内容, onOk: 确认回调, onCancel: 取消回调 |
| executeFunction | 执行自定义函数 | name: 函数名, params: 参数列表 |
| condition | 条件判断执行 | condition: 条件表达式, success: 成功执行动作, fail: 失败执行动作 |
| delay | 延迟执行 | time: 延迟时间(ms) |

## 事件参数传递

事件处理函数可以接收事件参数：

```json
{
  "methods": {
    "handleInputChange": {
      "type": "function",
      "args": ["value", "event"],
      "body": "state.formData.name = value;"
    }
  }
}
```

不同的组件事件会传入不同的参数，请参考各组件的文档了解具体参数。

## 事件冒泡与捕获

低代码平台的事件系统模拟了浏览器的事件冒泡机制：

```json
{
  "events": {
    "onClick": {
      "type": "function",
      "value": "handleClick",
      "stopPropagation": true
    }
  }
}
```

| 属性 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| stopPropagation | boolean | false | 是否阻止事件冒泡 |
| preventDefault | boolean | false | 是否阻止默认行为 |
| capture | boolean | false | 是否在捕获阶段触发 |

## 事件代理

对于列表渲染的场景，可以使用事件代理优化性能：

```json
{
  "componentName": "List",
  "props": {
    "dataSource": "{{state.items}}"
  },
  "events": {
    "onItemClick": {
      "type": "function",
      "value": "handleItemClick",
      "useEventDelegate": true
    }
  }
}
```

## 最佳实践

1. **职责分离**：事件处理函数应专注于业务逻辑，UI 更新通过状态变更自动触发
2. **避免过度嵌套**：复杂逻辑应拆分为多个小函数而非一个大函数
3. **状态管理**：使用 setState 动作而非直接修改 DOM
4. **错误处理**：在事件处理中添加适当的错误处理机制
5. **节流与防抖**：对于高频触发的事件（如滚动、输入），应用节流或防抖处理
6. **组件通信**：使用状态提升或发布-订阅模式实现组件间通信

## 调试技巧

1. 使用 `console.log` 在事件处理函数中输出调试信息
2. 低代码平台的开发者工具可以监控事件触发和状态变更
3. 设置断点调试复杂事件处理逻辑

## 兼容性说明

事件系统设计兼容主流浏览器，包括 Chrome、Firefox、Safari 和 Edge 的最新版本。
