# Input 输入框组件

输入框组件允许用户输入和编辑文本。

## 组件属性

| 属性名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| value | string | '' | 输入框的值 |
| placeholder | string | '' | 输入框占位文本 |
| type | string | 'text' | 输入框类型，可选值：'text', 'password', 'number', 'email' |
| disabled | boolean | false | 是否禁用 |
| readonly | boolean | false | 是否只读 |
| maxLength | number | null | 最大输入长度 |
| prefix | string | null | 前缀图标名称 |
| suffix | string | null | 后缀图标名称 |
| allowClear | boolean | false | 是否显示清除按钮 |
| autoFocus | boolean | false | 是否自动获取焦点 |

## 事件

| 事件名 | 说明 | 参数 |
|-------|------|------|
| onChange | 输入内容变化时触发 | value: string, event: Event |
| onFocus | 获取焦点时触发 | event: Event |
| onBlur | 失去焦点时触发 | event: Event |
| onPressEnter | 按下回车键时触发 | event: Event |
| onClear | 点击清除按钮时触发 | event: Event |

## 使用示例

```json
{
  "componentName": "Input",
  "props": {
    "placeholder": "请输入用户名",
    "type": "text",
    "maxLength": 50,
    "allowClear": true,
    "prefix": "user"
  },
  "events": {
    "onChange": {
      "type": "function",
      "value": "handleUsernameChange"
    }
  }
}
```

## 校验和集成

Input 组件通常与表单验证系统集成，可以通过以下方式添加校验：

```json
{
  "componentName": "Input",
  "props": {
    "placeholder": "请输入邮箱"
  },
  "validation": {
    "required": true,
    "type": "email",
    "message": "请输入有效的邮箱地址"
  }
}
```

## 最佳实践

1. 为输入框提供明确的 placeholder，指导用户输入
2. 根据输入内容类型选择合适的 input type
3. 适当使用前缀和后缀图标提供视觉提示
4. 为重要字段添加适当的校验规则
