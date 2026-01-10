# Button 按钮组件

按钮组件是用户交互的基础元素，用于触发操作或事件。

## 组件属性

| 属性名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| text | string | '' | 按钮文本内容 |
| type | string | 'default' | 按钮类型，可选值：'primary', 'default', 'danger', 'link' |
| size | string | 'medium' | 尺寸，可选值：'large', 'medium', 'small' |
| disabled | boolean | false | 是否禁用 |
| icon | string | null | 按钮图标，接受图标名称 |
| loading | boolean | false | 是否显示加载状态 |
| block | boolean | false | 是否为块级元素 |

## 事件

| 事件名 | 说明 | 参数 |
|-------|------|------|
| onClick | 点击按钮时触发 | event: Event |

## 使用示例

```json
{
  "componentName": "Button",
  "props": {
    "text": "提交",
    "type": "primary",
    "size": "medium",
    "disabled": false
  },
  "events": {
    "onClick": {
      "type": "function",
      "value": "submitForm"
    }
  }
}
```

## 最佳实践

1. 使用不同类型的按钮区分主次操作
2. 为按钮添加适当的文本描述操作
3. 在表单提交等场景中，考虑使用 loading 状态提供用户反馈
4. 危险操作应使用 'danger' 类型并考虑添加二次确认
