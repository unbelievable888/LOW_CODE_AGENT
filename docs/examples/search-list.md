# 如何实现一个搜索列表

本文档通过实例讲解如何使用低代码平台构建一个带搜索功能的列表页面，包含搜索条件、数据列表和分页功能。

## 页面结构

一个完整的搜索列表页面通常包含以下部分：

1. 搜索区域：提供多种搜索条件
2. 操作区域：包含批量操作和添加按钮
3. 数据表格：展示搜索结果
4. 分页控件：控制数据分页

## 实现步骤

### 第一步：创建页面结构

首先，我们创建基本的页面结构，包括搜索区域和数据表格：

```json
{
  "componentName": "Page",
  "title": "产品列表",
  "props": {
    "style": {
      "padding": "24px"
    }
  },
  "children": [
    {
      "componentName": "Card",
      "props": {
        "title": "搜索条件",
        "style": {
          "marginBottom": "24px"
        }
      },
      "children": [
        // 搜索表单将在这里定义
      ]
    },
    {
      "componentName": "Card",
      "props": {
        "title": "产品列表",
        "style": {
          "marginBottom": "24px"
        }
      },
      "children": [
        // 数据表格将在这里定义
      ]
    }
  ]
}
```

### 第二步：定义页面状态

接下来，我们定义页面状态，包括搜索条件、表格数据和分页信息：

```json
{
  "state": {
    "searchForm": {
      "name": "",
      "category": "",
      "status": "",
      "priceRange": [null, null]
    },
    "tableData": {
      "list": [],
      "total": 0,
      "loading": false
    },
    "pagination": {
      "current": 1,
      "pageSize": 10,
      "showSizeChanger": true,
      "showTotal": true
    },
    "selectedRows": [],
    "categories": []
  }
}
```

### 第三步：设计搜索表单

使用 FormContainer 组件构建搜索表单：

```json
{
  "componentName": "FormContainer",
  "props": {
    "name": "searchForm",
    "layout": "inline",
    "initialValues": "{{state.searchForm}}"
  },
  "children": [
    {
      "componentName": "FormItem",
      "props": {
        "name": "name",
        "label": "产品名称"
      },
      "children": [
        {
          "componentName": "Input",
          "props": {
            "placeholder": "请输入产品名称",
            "allowClear": true
          }
        }
      ]
    },
    {
      "componentName": "FormItem",
      "props": {
        "name": "category",
        "label": "产品分类"
      },
      "children": [
        {
          "componentName": "Select",
          "props": {
            "placeholder": "请选择产品分类",
            "allowClear": true,
            "options": "{{state.categories}}"
          }
        }
      ]
    },
    {
      "componentName": "FormItem",
      "props": {
        "name": "status",
        "label": "产品状态"
      },
      "children": [
        {
          "componentName": "Select",
          "props": {
            "placeholder": "请选择产品状态",
            "allowClear": true,
            "options": [
              { "label": "已上架", "value": "online" },
              { "label": "已下架", "value": "offline" },
              { "label": "待审核", "value": "pending" }
            ]
          }
        }
      ]
    },
    {
      "componentName": "FormItem",
      "props": {
        "name": "priceRange",
        "label": "价格范围"
      },
      "children": [
        {
          "componentName": "RangePicker",
          "props": {
            "type": "number",
            "placeholder": ["最低价格", "最高价格"]
          }
        }
      ]
    },
    {
      "componentName": "FormItem",
      "props": {},
      "children": [
        {
          "componentName": "Space",
          "children": [
            {
              "componentName": "Button",
              "props": {
                "text": "搜索",
                "type": "primary",
                "icon": "search"
              },
              "events": {
                "onClick": {
                  "type": "function",
                  "value": "handleSearch"
                }
              }
            },
            {
              "componentName": "Button",
              "props": {
                "text": "重置",
                "icon": "reload"
              },
              "events": {
                "onClick": {
                  "type": "function",
                  "value": "handleReset"
                }
              }
            }
          ]
        }
      ]
    }
  ]
}
```

### 第四步：设计数据表格

使用 Table 组件展示数据列表：

```json
{
  "componentName": "Flex",
  "props": {
    "justify": "space-between",
    "style": {
      "marginBottom": "16px"
    }
  },
  "children": [
    {
      "componentName": "Space",
      "children": [
        {
          "componentName": "Button",
          "props": {
            "text": "批量删除",
            "type": "primary",
            "danger": true,
            "disabled": "{{state.selectedRows.length === 0}}",
            "icon": "delete"
          },
          "events": {
            "onClick": {
              "type": "function",
              "value": "handleBatchDelete"
            }
          }
        },
        {
          "componentName": "Button",
          "props": {
            "text": "批量导出",
            "disabled": "{{state.selectedRows.length === 0}}",
            "icon": "export"
          },
          "events": {
            "onClick": {
              "type": "function",
              "value": "handleBatchExport"
            }
          }
        }
      ]
    },
    {
      "componentName": "Button",
      "props": {
        "text": "新增产品",
        "type": "primary",
        "icon": "plus"
      },
      "events": {
        "onClick": {
          "type": "function",
          "value": "handleAdd"
        }
      }
    }
  ]
}
```

接下来，定义数据表格：

```json
{
  "componentName": "Table",
  "props": {
    "rowKey": "id",
    "columns": [
      {
        "title": "产品图片",
        "dataIndex": "image",
        "render": {
          "type": "component",
          "component": "Image",
          "props": {
            "src": "{{record.image}}",
            "width": 60,
            "height": 60
          }
        }
      },
      {
        "title": "产品名称",
        "dataIndex": "name",
        "sorter": true
      },
      {
        "title": "产品分类",
        "dataIndex": "category",
        "filters": "{{state.categories.map(item => ({text: item.label, value: item.value}))}}"
      },
      {
        "title": "价格",
        "dataIndex": "price",
        "sorter": true,
        "render": {
          "type": "component",
          "component": "Text",
          "props": {
            "children": "¥ {{record.price.toFixed(2)}}"
          }
        }
      },
      {
        "title": "库存",
        "dataIndex": "stock",
        "sorter": true
      },
      {
        "title": "状态",
        "dataIndex": "status",
        "render": {
          "type": "component",
          "component": "Tag",
          "props": {
            "color": "{{record.status === 'online' ? 'green' : (record.status === 'offline' ? 'red' : 'orange')}}",
            "children": "{{record.status === 'online' ? '已上架' : (record.status === 'offline' ? '已下架' : '待审核')}}"
          }
        },
        "filters": [
          {
            "text": "已上架",
            "value": "online"
          },
          {
            "text": "已下架",
            "value": "offline"
          },
          {
            "text": "待审核",
            "value": "pending"
          }
        ]
      },
      {
        "title": "创建时间",
        "dataIndex": "createTime",
        "sorter": true
      },
      {
        "title": "操作",
        "key": "action",
        "render": {
          "type": "component",
          "component": "Space",
          "children": [
            {
              "componentName": "Button",
              "props": {
                "text": "编辑",
                "type": "link",
                "size": "small"
              },
              "events": {
                "onClick": {
                  "type": "function",
                  "params": ["{{record}}"],
                  "value": "handleEdit"
                }
              }
            },
            {
              "componentName": "Divider",
              "props": {
                "type": "vertical"
              }
            },
            {
              "componentName": "Button",
              "props": {
                "text": "删除",
                "type": "link",
                "danger": true,
                "size": "small"
              },
              "events": {
                "onClick": {
                  "type": "actionFlow",
                  "value": [
                    {
                      "type": "confirmDialog",
                      "params": {
                        "title": "确认删除",
                        "content": "确定要删除这个产品吗？此操作不可撤销。",
                        "okText": "确认",
                        "cancelText": "取消",
                        "onOk": {
                          "type": "function",
                          "params": ["{{record}}"],
                          "value": "handleDelete"
                        }
                      }
                    }
                  ]
                }
              }
            }
          ]
        }
      }
    ],
    "dataSource": "{{state.tableData.list}}",
    "loading": "{{state.tableData.loading}}",
    "pagination": {
      "current": "{{state.pagination.current}}",
      "pageSize": "{{state.pagination.pageSize}}",
      "total": "{{state.tableData.total}}",
      "showSizeChanger": "{{state.pagination.showSizeChanger}}",
      "showTotal": {
        "type": "function",
        "value": "total => `共 ${total} 条记录`"
      }
    },
    "rowSelection": {
      "type": "checkbox",
      "selectedRowKeys": "{{state.selectedRows.map(item => item.id)}}",
      "onChange": {
        "type": "function",
        "value": "handleSelectionChange"
      }
    }
  },
  "events": {
    "onChange": {
      "type": "function",
      "value": "handleTableChange"
    }
  }
}
```

### 第五步：配置数据源

为搜索列表配置数据源：

```json
{
  "dataSource": {
    "fetchProducts": {
      "type": "api",
      "options": {
        "url": "/api/products",
        "method": "GET",
        "params": {
          "name": "{{state.searchForm.name}}",
          "category": "{{state.searchForm.category}}",
          "status": "{{state.searchForm.status}}",
          "minPrice": "{{state.searchForm.priceRange?.[0]}}",
          "maxPrice": "{{state.searchForm.priceRange?.[1]}}",
          "page": "{{state.pagination.current}}",
          "pageSize": "{{state.pagination.pageSize}}",
          "sorter": "{{state.sorter?.field}}",
          "order": "{{state.sorter?.order}}"
        }
      },
      "willFetch": {
        "type": "function",
        "args": [],
        "body": "this.state.tableData.loading = true;"
      },
      "onSuccess": {
        "type": "function",
        "args": ["res"],
        "body": "this.state.tableData.list = res.data; this.state.tableData.total = res.total; this.state.tableData.loading = false;"
      },
      "onError": {
        "type": "function",
        "args": ["error"],
        "body": "this.state.tableData.loading = false; console.error(error);"
      }
    },
    "fetchCategories": {
      "type": "api",
      "options": {
        "url": "/api/categories",
        "method": "GET"
      },
      "autoFetch": true,
      "onSuccess": {
        "type": "function",
        "args": ["res"],
        "body": "this.state.categories = res.map(item => ({ label: item.name, value: item.id }));"
      }
    },
    "deleteProduct": {
      "type": "api",
      "options": {
        "url": "/api/products/:id",
        "method": "DELETE"
      },
      "onSuccess": {
        "type": "function",
        "args": [],
        "body": "this.dataSource.fetchProducts.load();"
      }
    },
    "batchDeleteProducts": {
      "type": "api",
      "options": {
        "url": "/api/products/batch-delete",
        "method": "POST",
        "data": {
          "ids": "{{state.selectedRows.map(item => item.id)}}"
        }
      },
      "onSuccess": {
        "type": "function",
        "args": [],
        "body": "this.state.selectedRows = []; this.dataSource.fetchProducts.load();"
      }
    }
  }
}
```

### 第六步：实现页面方法

实现各种交互方法：

```json
{
  "methods": {
    "handleSearch": {
      "type": "function",
      "args": [],
      "body": "this.state.pagination.current = 1; this.dataSource.fetchProducts.load();"
    },
    "handleReset": {
      "type": "function",
      "args": [],
      "body": "this.state.searchForm = { name: '', category: '', status: '', priceRange: [null, null] }; this.state.pagination.current = 1; this.dataSource.fetchProducts.load();"
    },
    "handleTableChange": {
      "type": "function",
      "args": ["pagination", "filters", "sorter"],
      "body": "this.state.pagination.current = pagination.current; this.state.pagination.pageSize = pagination.pageSize; this.state.filters = filters; this.state.sorter = sorter; this.dataSource.fetchProducts.load();"
    },
    "handleSelectionChange": {
      "type": "function",
      "args": ["selectedRowKeys", "selectedRows"],
      "body": "this.state.selectedRows = selectedRows;"
    },
    "handleDelete": {
      "type": "function",
      "args": ["record"],
      "body": "this.dataSource.deleteProduct.load({ id: record.id });"
    },
    "handleBatchDelete": {
      "type": "function",
      "args": [],
      "body": "if (this.state.selectedRows.length === 0) return; this.dataSource.batchDeleteProducts.load();"
    },
    "handleEdit": {
      "type": "function",
      "args": ["record"],
      "body": "window.location.href = `/product/edit/${record.id}`;"
    },
    "handleAdd": {
      "type": "function",
      "args": [],
      "body": "window.location.href = '/product/add';"
    },
    "handleBatchExport": {
      "type": "function",
      "args": [],
      "body": "if (this.state.selectedRows.length === 0) return; const ids = this.state.selectedRows.map(item => item.id).join(','); window.open(`/api/products/export?ids=${ids}`);"
    }
  }
}
```

### 第七步：页面生命周期

配置页面生命周期函数：

```json
{
  "lifeCycles": {
    "didMount": {
      "type": "function",
      "args": [],
      "body": "this.dataSource.fetchProducts.load();"
    }
  }
}
```

## 完整实现效果

通过上述步骤，我们实现了一个完整的搜索列表页面，包括：

1. 多条件搜索表单
2. 支持排序和筛选的数据表格
3. 分页控制
4. 批量操作和单项操作
5. 数据加载和交互

这个搜索列表实现了以下功能：
- 按多个条件搜索产品
- 表格数据排序和筛选
- 分页浏览数据
- 单条记录的编辑和删除
- 批量删除和导出
- 添加新记录

## 扩展功能

可以根据实际需求扩展以下功能：

1. **高级搜索**：添加可展开的高级搜索区域，包含更多搜索条件
2. **表格自定义列**：允许用户自定义显示哪些列
3. **数据可视化**：添加图表展示数据分析结果
4. **数据导入**：支持批量导入数据
5. **列表视图切换**：支持表格、卡片等多种视图模式

## 性能优化

对于大数据量的列表页面，可以考虑以下优化：

1. **虚拟滚动**：对于大量数据，使用虚拟滚动技术
2. **搜索防抖**：对搜索输入添加防抖处理
3. **缓存策略**：缓存已加载的数据，减少重复请求
4. **按需加载**：延迟加载不在视口的内容

以上就是在低代码平台上实现一个搜索列表页面的完整教程。
