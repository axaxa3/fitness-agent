# 健身 Agent 设计文档

> 面向大众用户的 AI 健身训练助手，基于 LangChain/LangGraph + FastAPI + Milvus + MongoDB

**日期**: 2026-07-16
**状态**: 设计完成，待用户审阅

---

## 1. 产品概述

### 1.1 定位

面向大众用户的 AI 健身训练 Web 应用。覆盖健身新手到进阶用户，提供个性化训练计划生成、训练记录追踪、动态计划调整和健身知识问答。

### 1.2 核心功能

| 功能模块 | 说明 |
|----------|------|
| 对话引导 | Agent 像真人教练一样聊着收集用户信息，生成用户画像 |
| 训练计划生成 | 多 Agent 协作（分化设计/动作选择/负荷规划/安全审核），生成个性化训练计划 |
| 训练执行记录 | 训练中逐组快速记录（组数、次数、重量、RPE），训练后对话补充感受 |
| 动态计划调整 | 每次训练后多 Agent 复盘（进度/疲劳/动作质量/调整建议），微调下次训练 |
| 自由问答 | 基于 Milvus 知识库的 RAG 问答，覆盖训练科学、动作要点等 |

### 1.3 不做什么（MVP）

- 饮食营养管理（后续版本）
- 睡眠追踪和恢复管理（后续版本）
- 体态评估（后续版本）
- 可穿戴设备接入（后续版本）
- 社交功能
- 原生移动 App（Web 优先）

---

## 2. 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI + Uvicorn | Python 全栈 |
| Agent 编排 | LangChain + LangGraph | StateGraph 工作流编排 |
| LLM | 阿里百炼/千问（OpenAI 兼容 API） | ChatOpenAI 统一入口 |
| 向量数据库 | Milvus | 动作语义搜索 + 健身知识库 + 长期记忆 |
| 文档数据库 | MongoDB | 用户画像、训练计划、训练日志、记忆摘要 |
| 嵌入模型 | DashScope text-embedding-v4（1024维） | 稠密向量 |
| 前端 | 原生 HTML/CSS/JS | 无构建工具，CDN 引入 marked/highlight.js |
| 流式输出 | SSE（FastAPI StreamingResponse） | 对话和多 Agent 进度推送 |
| 包管理 | uv | 依赖管理 |
| 日志 | Loguru | 结构化日志 |
| 配置 | Pydantic Settings | .env 环境变量 |
| Python | >= 3.11 | - |

**与现有项目（OnCall / 掌柜智库）技术栈完全一致。**

---

## 3. 架构设计

### 3.1 整体架构

```
┌──────────────────────────────────────────────────┐
│              前端 (static/)                        │
│         原生 HTML/CSS/JS + SSE 流式                │
└────────────────────┬─────────────────────────────┘
                     │ HTTP + SSE
┌────────────────────▼─────────────────────────────┐
│              FastAPI 应用                          │
│                                                   │
│  ┌──────────┬──────────┬──────────┬───────────┐  │
│  │onboarding│   plan   │ training │    chat    │  │
│  │ 单Agent  │ 多Agent  │ 多Agent  │  单Agent   │  │
│  │ 引导对话 │ 计划生成 │ 执行复盘 │  自由问答  │  │
│  └────┬─────┴────┬─────┴────┬─────┴─────┬─────┘  │
│       │          │          │           │        │
│  ┌────┴──────────┴──────────┴───────────┴────┐   │
│  │              共享工具层                     │   │
│  │  动作语义搜索(Milvus) | 训练数据分析       │   │
│  └────────────────────────────────────────────┘   │
│                                                   │
│  ┌────────────────────────────────────────────┐   │
│  │              记忆管理                        │   │
│  │  四层：短期→中期→长期→用户画像               │   │
│  └────────────────────────────────────────────┘   │
└────────────────────┬─────────────────────────────┘
                     │
    ┌────────────────┼────────────────┐
    ▼                ▼                ▼
┌───────┐    ┌──────────┐    ┌──────────┐
│Milvus │    │ MongoDB  │    │  LLM API │
│向量DB │    │ 文档存储  │    │ 百炼千问  │
└───────┘    └──────────┘    └──────────┘
```

### 3.2 四大场景模块

**按用户场景拆分，非按技术层拆分。每个模块独立 LangGraph + 独立 API。**

| 模块 | Agent 类型 | 核心职责 |
|------|-----------|----------|
| `onboarding/` | 单 Agent | 对话收集用户信息 → 生成用户画像（完成后自动触发计划生成） |
| `plan/` | 多 Agent 协作 | 分化设计 + 动作选择 + 负荷规划 + 安全审核 → 生成完整计划 |
| `training/` | 多 Agent 协作 | 训练记录 + 进度分析 + 疲劳监测 + 动作评估 + 计划微调 |
| `chat/` | 单 Agent + RAG | 自由问答（动作搜索 + 知识库检索增强） |

**设计原则**：每个模块状态结构简单、节点少、好调试、可独立迭代。

### 3.3 多 Agent 协作设计

#### 计划生成 — "教练组会诊"

```
用户画像
    │
    ▼
[Supervisor 主教练]
    │
    ├─→ [分化设计师]     决定训练分化方式(PPL/UpperLower/FullBody)
    ├─→ [动作专家]       按肌群选最佳动作组合，确保均衡不重复
    ├─→ [负荷规划师]     设定组数/次数/重量/RPE，渐进方案
    └─→ [安全审核员]     检查伤病史，排除禁忌动作，验证难度合理
    │
    ▼
[Supervisor 汇总] → 最终训练计划 + 决策理由
```

#### 训练复盘 — "赛后复盘组"

```
训练完成数据
    │
    ├─→ [进度分析师]     目标肌群训练量达标？力量在涨？需加容量？
    ├─→ [疲劳监测员]     连续RPE偏高？该安排减载？
    ├─→ [动作质量评估]   用户反馈伤病 → 替换动作建议
    └─→ [下次训练调整]   综合修改下次训练的动作/组数/重量
    │
    ▼
[Supervisor 汇总] → 调整建议 + 更新用户画像
```

---

## 4. 四层记忆架构

```
Layer 1 ─ 短期：对话窗口（内存）
  ├─ 最近 10 条交互完整消息
  ├─ 场景上下文注入（当前训练、用户画像摘要）
  └─ 生命周期：单次对话

Layer 2 ─ 中期：滚动摘要（MongoDB）
  ├─ 按主题分桶：训练反馈/伤病关注/用户偏好/计划迭代/问答模式
  ├─ 按场景选择性注入（训练场景注训练桶，计划场景注偏好和迭代桶）
  └─ 生命周期：几周（旧摘要转长期记忆后被归约）

Layer 3 ─ 长期：语义记忆（Milvus session_memory）
  ├─ 摘要片段做 embedding
  ├─ LLM 去重 + 重要性评分 → 过阈值才写入
  ├─ 对话时按需检索最相关历史
  └─ 生命周期：永久

Layer 4 ─ 用户画像：结构化档案（MongoDB user_profiles）
  ├─ 基础信息、目标、水平、器械、频率
  ├─ 力量快照（各动作估算 1RM）
  ├─ 肌群平衡评估
  ├─ 伤病标记（含限制动作列表）
  ├─ 疲劳趋势
  └─ 每次训练后自动更新相关字段
```

---

## 5. 项目目录结构

```
fitness-agent/
├── .env
├── pyproject.toml
├── uv.lock
├── app/
│   ├── main.py
│   ├── conf/
│   │   ├── settings.py
│   │   ├── llm_config.py
│   │   └── db_config.py
│   ├── core/
│   │   ├── logger.py
│   │   ├── llm_factory.py
│   │   └── prompt_loader.py
│   ├── data/
│   │   ├── exercise_library.py
│   │   ├── user_profile.py
│   │   ├── training_log.py
│   │   ├── memory_manager.py
│   │   ├── memory_milvus.py
│   │   └── mongo_client.py
│   ├── onboarding/
│   │   ├── api.py
│   │   ├── graph.py
│   │   └── state.py
│   ├── plan/
│   │   ├── api.py
│   │   ├── supervisor.py
│   │   ├── agents/
│   │   │   ├── split_designer.py
│   │   │   ├── exercise_selector.py
│   │   │   ├── volume_planner.py
│   │   │   └── safety_checker.py
│   │   ├── graph.py
│   │   └── state.py
│   ├── training/
│   │   ├── api.py
│   │   ├── supervisor.py
│   │   ├── agents/
│   │   │   ├── progress_analyzer.py
│   │   │   ├── fatigue_monitor.py
│   │   │   ├── quality_assessor.py
│   │   │   └── next_session_adjuster.py
│   │   ├── graph.py
│   │   └── state.py
│   ├── chat/
│   │   ├── api.py
│   │   ├── graph.py
│   │   └── state.py
│   └── tools/
│       ├── exercise_search.py
│       └── training_analyzer.py
├── prompts/
│   ├── onboarding.prompt
│   ├── plan_generate.prompt
│   ├── plan_adjust.prompt
│   ├── chat.prompt
│   └── memory_summary.prompt
├── seeds/
│   └── exercise_seed.json
└── static/
    ├── index.html
    ├── css/
    └── js/
```

---

## 6. API 设计

| 方法 | 路径 | 说明 | 流式 |
|------|------|------|:---:|
| `POST` | `/api/onboard/start` | 初始化引导会话 | - |
| `POST` | `/api/onboard/message` | 引导对话交互 | SSE |
| `POST` | `/api/plan/generate` | 多 Agent 生成计划 | SSE |
| `GET` | `/api/plan/current` | 获取当前计划 | - |
| `POST` | `/api/plan/adjust` | 手动请求调整计划 | SSE |
| `GET` | `/api/train/today` | 获取今日训练 | - |
| `POST` | `/api/train/log` | 记录单组数据 | - |
| `POST` | `/api/train/complete` | 完成训练+复盘 | SSE |
| `POST` | `/api/chat/message` | 自由问答 | SSE |
| `GET` | `/api/exercise/search` | 动作语义搜索 | - |
| `GET` | `/api/profile` | 用户画像 | - |
| `GET` | `/api/history` | 训练历史 | - |
| `GET` | `/api/health` | 健康检查 | - |

---

## 7. 数据库 Schema

### 7.1 MongoDB 集合

| 集合 | 说明 |
|------|------|
| `user_profiles` | 用户画像（基础信息、健身档案、力量快照、伤病标记、疲劳趋势） |
| `training_plans` | 训练计划（周期、周次、分化方式、每次训练的练习列表含组数/次数/重量/RPE） |
| `training_logs` | 训练日志（每组记录含组数/次数/重量/RPE、整体感受、调整建议） |
| `memory_summaries` | 中期记忆摘要（按主题分桶：训练反馈/伤病/偏好/计划迭代/问答模式） |

### 7.2 Milvus 集合

| 集合 | 说明 | 向量来源 |
|------|------|----------|
| `exercise_kb` | 动作语义库（120条动作描述文本） | text-embedding-v4 |
| `fitness_knowledge` | 健身知识库（训练科学文章分块） | text-embedding-v4 |
| `session_memory` | 长期语义记忆（按用户 ID 分区，含重要性评分） | text-embedding-v4 |

### 7.3 动作库字段设计（exercise_kb）

| 字段 | 类型 | 说明 |
|------|------|------|
| `exercise_id` | varchar | 唯一 ID |
| `name` | varchar | 动作名称（中英文） |
| `name_cn` | varchar | 中文名 |
| `text` | varchar | 语义搜索文本（描述+要点+肌群） |
| `primary_muscles` | json | 主要目标肌群（多标签） |
| `secondary_muscles` | json | 次要肌群 |
| `equipment` | varchar | 所需器械 |
| `difficulty` | int | 难度 1-3 |
| `category` | varchar | 分类：compound/isolation/bodyweight |
| `rep_range` | json | 适合次数范围 [min, max] |
| `unilateral` | bool | 是否单侧动作 |
| `alternatives` | json | 替代动作 ID 列表 |
| `tips` | varchar | 简短动作要点 |
| `contraindications` | json | 禁忌场景（如"腰椎问题"） |
| `vector` | float[1024] | text 字段的稠密向量 |

---

## 8. SSE 流式机制

沿用 RAG-project1 的成熟方案：

```
POST /api/xxx/message
    │  创建 session_id + queue.Queue
    │  启动后台任务 run_graph()
    ▼
LangGraph 节点执行中
    │  push_to_session(event)
    │  event: { type: "progress"|"delta"|"final"|"error", data: {...} }
    ▼
GET /api/xxx/stream/{session_id}
    │  读取 queue，yield SSE 格式
    │  客户端断连 → 自动清理资源
    ▼
SSE 输出：
  event: progress     → 多 Agent 进度（"分化设计师正在分析..."）
  event: delta        → 流式 token（LLM 逐词输出）
  event: final        → 完整结果 + 关闭连接
  event: error        → 错误信息
```

---

## 9. 开发顺序

| 阶段 | 内容 | 依赖 |
|------|------|------|
| P0 | 项目骨架：目录结构、配置、LLM Factory、MongoDB/Milvus 客户端 | - |
| P1 | 动作库种子数据 + 动作搜索工具 | P0 |
| P2 | `onboarding/` 引导对话 + 用户画像 CRUD | P0 |
| P3 | `plan/` 多 Agent 计划生成 | P1, P2 |
| P4 | `training/` 训练记录 + 多 Agent 复盘 | P2, P3 |
| P5 | `chat/` RAG 自由问答 + 健身知识库 | P1 |
| P6 | 记忆管理：四层记忆系统 | P2, P4, P5 |
| P7 | 前端页面：引导页、计划页、训练页、聊天页 | P2-P6 |
| P8 | 集成测试 + 体验打磨 | P7 |

---

## 10. Key Design Decisions

1. **按场景拆分而非按技术层拆分**：四个独立 LangGraph 工作流，各自有清晰的职责边界
2. **多 Agent 仅用于需要"多视角"的场景**：计划生成和训练复盘，而非所有场景
3. **四层记忆而非单层**：适配健身场景"时间跨度极大"的特点（对话→训练→周期→画像）
4. **动作库在 Milvus 做语义搜索而非仅结构化过滤**：支持"有没有不伤膝盖的练腿动作"这类模糊查询
5. **纯原生前端**：与现有项目风格一致，零构建工具，SSE + queue.Queue 实现流式
6. **纯独立系统**：MVP 不依赖任何第三方 API（手表、健身 App 等）
