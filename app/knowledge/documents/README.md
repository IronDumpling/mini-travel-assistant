# 知识库文档目录

此目录用于存储旅游相关的知识文档。

## 目录结构

```
documents/
├── destinations/           # 目的地信息
│   ├── asia/              # 亚洲
│   ├── europe/            # 欧洲
│   ├── america/           # 美洲
│   └── others/            # 其他地区
├── transportation/         # 交通信息
│   ├── airlines/          # 航空公司
│   ├── trains/            # 火车
│   └── buses/             # 巴士
├── accommodation/          # 住宿信息
│   ├── hotels/            # 酒店
│   ├── hostels/           # 青年旅社
│   └── bnb/               # 民宿
├── activities/            # 活动体验
│   ├── sightseeing/       # 观光
│   ├── adventure/         # 冒险
│   └── cultural/          # 文化
└── practical/             # 实用信息
    ├── visa/              # 签证
    ├── currency/          # 货币
    ├── weather/           # 天气
    └── safety/            # 安全
```

## 文档格式

支持的文档格式：
- JSON (.json)
- YAML (.yaml/.yml)
- Markdown (.md)
- 纯文本 (.txt)

## 知识文档模板

```json
{
    "id": "unique_knowledge_id",
    "title": "知识标题",
    "content": "详细内容...",
    "category": "destinations",
    "location": "具体位置",
    "tags": ["标签1", "标签2"],
    "language": "zh",
    "source": {
        "name": "数据来源",
        "url": "https://example.com",
        "reliability_score": 0.9
    },
    "last_updated": "2024-01-01T00:00:00Z"
}
``` 