# SQL 生成规则说明（可扩展提示词）

本文件用于描述/新增 SQL 生成规则，供后续自动化或人工更新 `internal/query/querybuilder.go` 的生成逻辑时参考。建议每条规则都包含：

- 适用模型：例如 Skew/Holes/LowCard（内部模型），或指定表名。
- 目标列：涉及到的列（如 `<Table>_int`、`<Table>_varchar`、`<Table>_datetime`）。
- 查询意图/模式：point lookup / range scan / histogram 边界探测 / out of bound / 组合条件 等。
- 标签（Query Label）：用于报告与汇总（HTML/CSV/JSON）的唯一标识，便于分组与对比。
- 生成细节：如使用当前统计信息的最小/最大值，或使用“初始化基线（Base）”阶段的 min/max，取样数量、步进、是否去重、是否避免连续值等。

---

## 现有内置规则（摘要）

下述规则在 `internal/query/querybuilder.go` 中实现，针对每个内部模型表（以 `<T>` 表示表名前缀，如 `Skew`、`Holes`、`LowCard`）：

- 整数列 `<T>_int`
  - LABEL: `int out of bound`，等值命中到“当前 maxInt+1000”
  - LABEL: `int point lookup`，等值命中到“当前 minInt+1”
  - LABEL: `int range scan`，区间 `[minInt, minInt+50]`
  - LABEL: `int last value in last histogram`，等值命中“初始化基线 Base 的最大值（baseMaxInt）`
  - LABEL: `int first value in first histogram`，等值命中“初始化基线 Base 的最小值（baseMinInt）`
  - 模型为 Holes 时，额外生成：
    - LABEL: `int in the hole`，条件落在洞区间 (holeStart, holeEnd)
    - LABEL: `int across hole`，跨越洞边界的区间（holeStart±offset）
    - LABEL: `int include hole`，条件包含时间洞区间 (holeStart-offset, holeEnd+offset)

- 字符串列 `<T>_varchar`（prefix+suffixRange）
  - LABEL: `string out of bound`，等值命中到 `prefix + (suffixEnd+1000)`
  - LABEL: `string point lookup`，等值命中到 `prefix + (suffixStart+1)`
  - LABEL: `string range scan`，区间 `[prefix+suffixStart, prefix+(suffixStart+50)]`
  - LABEL: `string last value in last histogram`，等值命中 `prefix + suffixEnd`（使用 Base 的边界）
  - LABEL: `string first value in first histogram`，等值命中 `prefix + suffixStart`（使用 Base 的边界）

- 日期时间列 `<T>_datetime`
  - LABEL: `datetime out of bound`，使用 `> dateMax`（守护非法最大值，取 Base 的最大日期）
  - LABEL: `datetime point lookup`，等值命中到 `dateMin` 附近的某一天
  - LABEL: `datetime range scan`，区间 `[dateMin, dateMin+30days]`
  - LABEL: `datetime last value in last histogram`，等值命中 Base 的最大日期（baseDateMax）
  - LABEL: `datetime first value in first histogram`，等值命中 Base 的最小日期（baseDateMin）
  - 模型为 Holes 时，额外生成：
    - LABEL: `datetime in the hole`，条件落在时间洞区间 (dhStart, dhEnd)
    - LABEL: `datetime across hole`，跨越洞边界（dhStart±offset）
    - LABEL: `datetime include hole`，条件包含时间洞区间 (dhStart-offset, dhEnd+offset) 

- 组合/混合
  - LABEL: `mixed condition`，形如：`<T>_int > minInt AND <T>_varchar LIKE 'prefix%'`
  - LABEL: `lots of IN`（已实现），a IN (...) AND b IN (...) AND c IN (...)，详见下方“初始化提示词示例”。

---

## 规则书写模板（建议）

```
规则名（Query Label）：<简明唯一的名称>
适用模型：<Skew|Holes|LowCard|ALL>
涉及列：<T>_int / <T>_varchar / <T>_datetime（可多列）
SQL 形态：<等值|范围|IN|LIKE|组合条件等，给出伪代码或样例>
取值来源：<当前统计 min/max | 初始化 Base min/max | 配置参数 | 常量>
参数/限制：<去重/非连续/数量上限/步长/容错等>
备注：<可选说明>
```

---

## 初始化提示词示例（lots of IN）

请为三个内部模型分别生成一个“lots of IN”的查询语句，要求：

- 形式：`a IN (...) AND b IN (...) AND c IN (...)`
- 其中 a,b,c 分别对应表的三列：`<T>_int`, `<T>_varchar`, `<T>_datetime`
- 三个 IN 列表的元素均为“非连续”的 1000 个值：
  - `<T>_int`：在 `[minInt, maxInt]` 之间，按步长≥2 取样，直到 1000 个；不足时按当前可得值补齐
  - `<T>_varchar`：基于 `prefix + [suffixStart, suffixEnd]`，按步长≥2 取样至 1000 个
  - `<T>_datetime`：在 `[dateMin, dateMax]` 的日粒度范围内按步长≥2 取样至 1000 个
- Query Label：`lots of IN`

（以上规则已在 `internal/query/querybuilder.go` 实现。如需修改/新增，请在本文件添加新的指令，或使用下述 YAML 结构化 DSL 以便自动注入。）

### 结构化 DSL（YAML）

支持通过 Markdown 中的 YAML fenced code block 声明规则，auto-run 会解析并尝试注入代码：

````yaml
rules:
  # Int 基础规则
  - label: "int out of bound"
    type: int_out_of_bound
  - label: "int range scan"
    type: int_range_scan
  - label: "int last value in last histogram"
    type: int_last_histogram
  - label: "int first value in first histogram"
    type: int_first_histogram

  # String 基础规则
  - label: "string out of bound"
    type: string_out_of_bound
  - label: "string point lookup"
    type: string_point_lookup
  - label: "string range scan"
    type: string_range_scan
  - label: "string last value in last histogram"
    type: string_last_histogram
  - label: "string first value in first histogram"
    type: string_first_histogram

  # Datetime 基础规则
  - label: "datetime out of bound"
    type: datetime_out_of_bound
  - label: "datetime point lookup"
    type: datetime_point_lookup
  - label: "datetime range scan"
    type: datetime_range_scan
  - label: "datetime last value in last histogram"
    type: datetime_last_histogram
  - label: "datetime first value in first histogram"
    type: datetime_first_histogram

  # Holes 专属规则（int & datetime）
  - label: "int in the hole"
    type: holes_int_in
  - label: "int across hole"
    type: holes_int_across
  - label: "int include hole"
    type: int_include_hole
    params:
      margin: 10
  - label: "datetime in the hole"
    type: holes_datetime_in
  - label: "datetime across hole"
    type: holes_datetime_across
  - label: "datetime include hole"
    type: datetime_include_hole   # 或 holes_datetime_include


  # 混合与综合
  - label: "mixed condition"
    type: mixed_condition
  - label: "lots of IN"
    type: lots_of_in
````

当前示例会在 Holes 模型中注入标签为 “int include hole” 的规则：

- SQL：`/* LABEL: int include hole */ SELECT ... WHERE <T>_int < (holeStart - margin) OR <T>_int > (holeEnd + margin)`
- 若该标签已存在，则不会重复注入。
