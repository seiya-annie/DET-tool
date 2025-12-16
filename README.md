# DET-Tool (Database Estimation Testing Tool)

DET-Tool 是一个数据库性能测试和查询优化分析工具，用于生成测试数据、执行查询并分析数据库统计信息的准确性。

## 功能特性

### 核心功能
- **数据生成**: 支持多种数据分布模式（倾斜、空洞、低基数）
- **查询生成**: 基于当前数据库状态自动生成测试查询
- **性能分析**: 执行查询并分析估算误差
- **报告生成**: 生成详细的CSV和HTML报告
- **外部基准测试**: 支持TPC-C和TPC-H基准测试

### 数据模型
1. **Skew (数据倾斜)**: 模拟数据分布不均匀的情况
2. **Holes (数据空洞)**: 模拟数据中存在空白区间的情况
3. **LowCard (低基数)**: 模拟不同值很少的情况
4. **External TPC-C**: 外部TPC-C基准测试
5. **External TPC-H**: 外部TPC-H基准测试

## 快速开始

### 前提条件
- Go 1.21 或更高版本
- MySQL/TiDB 数据库
- TiUP (用于外部基准测试)

### 安装

1. 克隆项目:
```bash
git clone <repository-url>
cd det-tool
```

2. 安装依赖:
```bash
go mod download
```

3. 构建项目:
```bash
go build -o det-tool
```

### 配置

#### 数据库配置 (db_config.json)
```json
{
  "host": "127.0.0.1",
  "port": 4000,
  "user": "root",
  "password": "",
  "db_name": "esti_test_db",
  "charset": "utf8mb4"
}
```

#### 模型配置 (config.json)
```json
{
  "models": [
    {
      "name": "Skew",
      "description": "数据倾斜模型",
      "type": "skew",
      "params": {
        "rows": 10000,
        "ndv": 1000,
        "skew_weights": [0.7, 0.2, 0.05],
        "int_range": [1, 10000],
        "date_range": ["2024-01-01", "2024-12-31"],
        "varchar_range": {"prefix": "user_", "suffix_range": [1, 10000]}
      },
      "incremental": {
        "insert_rows": 1000,
        "update_ratio": 0.2,
        "delete_ratio": 0.05
      }
    }
  ]
}
```

## 使用方法

### 基本命令

#### 1. 生成基础数据
```bash
./det-tool --gen-base
```

#### 2. 生成增量数据
```bash
./det-tool --gen-inc
```

#### 3. 生成查询
```bash
./det-tool --gen-query
```

#### 4. 执行查询并生成报告
```bash
./det-tool --exec-query
```

#### 5. 执行所有步骤
```bash
./det-tool --all
```

### 命令行参数

```
Flags:
      --all              Execute all steps: Base -> Query -> Inc -> Exec Query
      --config string    Configuration file for models (default "config.json")
      --db-config string Database configuration file (default "db_config.json")
      --exec-query       Step 4: Execute SQL Queries & Report
      --gen-base         Step 1: Generate & Load Base Data
      --gen-inc          Step 2: Generate & Execute Incremental Data
      --gen-query        Step 3: Generate SQL Queries (based on current DB stats)
  -h, --help             help for det-tool
      --sql-file string  File for incremental DML (default "incremental_dml.sql")
      --analyze-retries int         Max retries waiting for stats healthy after ANALYZE (default 20)
      --analyze-interval-ms int     Interval (ms) between retries waiting for stats healthy (default 1000)
```

## 数据模型详解

### Skew (数据倾斜)
模拟数据分布不均匀的情况，某些值出现频率远高于其他值。

参数说明:
- `rows`: 总行数
- `ndv`: 不同值的数量
- `skew_weights`: 倾斜权重分布
- `int_range`: 整数范围
- `date_range`: 日期范围
- `varchar_range`: 字符串范围

### Holes (数据空洞)
模拟数据中存在空白区间的情况。

参数说明:
- `int_hole_range`: 整数空洞范围
- `date_hole_range`: 日期空洞范围

### LowCard (低基数)
模拟不同值很少的情况。

参数说明:
- `ndv`: 不同值的数量 (通常很小)
- `varchar_range.options`: 预定义的选项列表

## 报告解读

### CSV报告
包含以下字段:
- Model: 模型名称
- stats_healthy_ratio: 统计信息健康度
- modify_ratio: 修改比例
- query_label: 查询标签（如：out of bound, point lookup, range scan, in the hole, across hole, mixed condition）
- estimation_error_ratio: 估算误差比例
- estimation_error_value: 估算误差值
- query: SQL查询
- duration_ms: 执行时间(毫秒)
- explain: 执行计划
- risk_operators_count: 风险操作符数量

### HTML报告
包含:
- 执行摘要统计
- 按模型分类的统计信息
- 详细的查询结果表格
- 风险查询高亮显示

## 外部基准测试

### TPC-C
TPC-C基准测试模拟完整的计算环境，包含5种事务类型，以每分钟事务数(tpmC)衡量性能。

### TPC-H
TPC-H基准测试包含一组面向业务的即席查询和并发数据修改，通过执行复杂查询来衡量决策支持系统的性能。

### 外部基准测试配置
```json
{
  "name": "TPCC_Bench",
  "type": "external_tpcc",
  "params": {
    "warehouses": 4,
    "threads": 4,
    "time": "1m"
  }
}
```

### 将 TPCC/TPCH 查询结果纳入报告
- 工具内置了一小组只读的 TPCC/TPCH 查询集合：
  - TPCH: 默认执行 q1/q6/q12，也可在 `incremental.queries` 中指定，如 `["q1","q6"]`
  - TPCC: 内置若干聚合/Join 查询，用于评估优化器行为
- 当运行 `--exec-query` 时，工具会：
  - 对内部模型(如 skew/holes/low_card)读取 `queries_*.sql` 执行
  - 对外部模型(external_tpcc/external_tpch)在其 `params.db_name` 指定的数据库上执行内置查询集合
- 所有查询结果(包括外部TPCC/TPCH)都会写入同一份 CSV/HTML/JSON 报告

## 高级用法

### 自定义查询生成
可以通过修改查询生成器来创建自定义的查询模式:

```go
queryBuilder := NewQueryBuilder()
queryBuilder.Generate(modelConfig, tableName, outputFile, currentStats)
```

### 数据生成扩展
可以通过扩展数据生成器来支持新的数据分布模式:

```go
dataGenerator := NewDataGenerator()
df := dataGenerator.Generate(modelConfig)
```

### 报告自定义
可以自定义报告生成器来创建特定格式的报告:

```go
reportGenerator := NewReportGenerator()
reportGenerator.GenerateCSVReport(results, filename, config)
reportGenerator.GenerateHTMLReport(results, filename, config)
```

## 性能优化建议

1. **数据库配置**:
   - 确保有足够的内存分配给数据库
   - 调整InnoDB缓冲池大小
   - 启用适当的索引

2. **批量操作**:
   - 使用批量插入代替单条插入
   - 合理设置批量大小

3. **统计信息**:
   - 定期更新统计信息
   - 监控统计信息健康度

4. **查询优化**:
   - 分析慢查询日志
   - 使用EXPLAIN分析查询计划

## 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查数据库配置
   - 确保数据库服务正在运行
   - 验证网络连接

2. **TiUP命令未找到**
   - 安装TiUP: `curl --proto '=https' --tlsv1.2 -sSf https://tiup-mirrors.pingcap.com/install.sh | sh`
   - 添加TiUP到PATH

3. **权限问题**
   - 确保数据库用户有足够的权限
   - 检查文件系统权限

4. **内存不足**
   - 减少数据量
   - 增加系统内存
   - 优化数据处理流程

### 调试模式
可以通过设置环境变量来启用详细日志:
```bash
export DET_DEBUG=1
./det-tool --all
```

## 贡献指南

欢迎贡献代码! 请遵循以下步骤:

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详情请见 [LICENSE](LICENSE) 文件。

## 联系方式

如有问题或建议，请通过以下方式联系:
- 邮箱: [your-email@example.com]
- 项目Issues: [项目Issues页面]

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 支持基本数据生成和查询执行
- 支持CSV和HTML报告生成
- 支持外部基准测试

### v1.1.0 (2024-02-01)
- 添加JSON报告格式
- 改进错误处理
- 优化性能
- 添加更多配置选项
