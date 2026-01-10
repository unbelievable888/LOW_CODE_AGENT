# FormContainer 表单容器组件

表单容器是管理表单元素的容器组件，提供数据收集、校验和提交功能。

## 组件属性

| 属性名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| name | string | 'form' | 表单名称，用于标识表单 |
| layout | string | 'vertical' | 表单布局，可选值：'vertical', 'horizontal', 'inline' |
| labelPosition | string | 'left' | 标签位置，可选值：'left', 'top', 'right' |
| labelWidth | number \| string | 'auto' | 标签宽度，可以是数字或 CSS 宽度值 |
| disabled | boolean | false | 是否禁用所有表单项 |
| initialValues | object | {} | 表单初始值 |
| validateOnMount | boolean | false | 是否在挂载时进行校验 |
| validateOnBlur | boolean | true | 是否在失焦时进行校验 |
| validateOnChange | boolean | false | 是否在值变化时进行校验 |
| scrollToError | boolean | false | 提交失败时是否滚动到第一个错误字段 |

## 事件

| 事件名 | 说明 | 参数 |
|-------|------|------|
| onSubmit | 提交表单且验证通过后触发 | values: object |
| onValuesChange | 字段值更新时触发 | changedValues: object, allValues: object |
| onValidate | 校验完成时触发 | errors: array \| null, values: object |
| onReset | 表单重置时触发 | event: Event |

## 方法

表单容器提供以下方法，可通过事件系统调用：

| 方法名 | 说明 | 参数 | 返回值 |
|-------|------|------|-------|
| submit | 提交表单 | - | Promise |
| validate | 校验表单 | fieldNames?: string[] | Promise |
| resetFields | 重置表单值为初始值 | fieldNames?: string[] | void |
| setFieldsValue | 设置表单字段值 | values: object | void |
| getFieldsValue | 获取表单字段值 | fieldNames?: string[] | object |

## 使用示例

```json
{
  "componentName": "FormContainer",
  "props": {
    "name": "loginForm",
    "layout": "vertical",
    "initialValues": {
      "remember": true
    }
  },
  "children": [
    {
      "componentName": "FormItem",
      "props": {
        "name": "username",
        "label": "用户名",
        "rules": [
          { "required": true, "message": "请输入用户名" }
        ]
      },
      "children": [
        {
          "componentName": "Input",
          "props": {
            "placeholder": "请输入用户名"
          }
        }
      ]
    },
    {
      "componentName": "FormItem",
      "props": {
        "name": "password",
        "label": "密码",
        "rules": [
          { "required": true, "message": "请输入密码" }
        ]
      },
      "children": [
        {
          "componentName": "Input",
          "props": {
            "type": "password",
            "placeholder": "请输入密码"
          }
        }
      ]
    },
    {
      "componentName": "FormItem",
      "children": [
        {
          "componentName": "Button",
          "props": {
            "type": "primary",
            "text": "登录",
            "block": true
          },
          "events": {
            "onClick": {
              "type": "function",
              "value": "submitLoginForm"
            }
          }
        }
      ]
    }
  ],
  "events": {
    "onSubmit": {
      "type": "function",
      "value": "handleLoginFormSubmit"
    }
  }
}
```

## 嵌套数据结构

表单容器支持复杂的嵌套数据结构：

```json
{
  "componentName": "FormContainer",
  "props": {
    "name": "userInfoForm"
  },
  "children": [
    {
      "componentName": "FormItem",
      "props": {
        "name": ["address", "city"],
        "label": "城市"
      },
      "children": [
        {
          "componentName": "Input"
        }
      ]
    }
  ]
}
```

## 最佳实践

1. 使用合理的表单布局提高用户体验
2. 为每个表单项添加适当的校验规则
3. 通过 initialValues 设置表单初始值
4. 使用 FormItem 组件包裹每个表单控件以便正确关联校验规则
5. 考虑表单项的分组和布局，提高大型表单的可用性
