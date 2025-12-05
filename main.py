import pandas as pd
import numpy as np
from faker import Faker
import json
import random
import argparse
import os
import sys
import copy
import time
import subprocess
import pymysql
from datetime import datetime, timedelta

fake = Faker()

# -----------------------------------------------------------
# 配置与常量
# -----------------------------------------------------------
INTERNAL_MODELS = ['skew', 'holes', 'low_card']
EXTERNAL_MODELS = ['external_tpcc', 'external_tpch']  # 纯外部工具
# TPCC/TPCH 如果配置为 internal (在 config.json 中 type=tpcc)，则走内部逻辑
TARGET_QUERY_MODELS = ['skew', 'holes', 'low_card']
CONTROL_KEYS = ['insert_rows', 'update_ratio', 'delete_ratio']


# ==========================================
# 1. 外部工具运行器
# ==========================================
class ExternalBenchRunner:
    def __init__(self, db_config):
        self.db = db_config

    def _build_base_cmd(self, tool_name):
        cmd = [
            "tiup", "bench", tool_name,
            "--host", self.db['host'],
            "--port", str(self.db['port']),
            "--user", self.db['user']
        ]
        if self.db.get('password'):
            cmd.extend(["--password", self.db['password']])
        if self.db.get('db_name'):
            cmd.extend(["--db", self.db['db_name']])
        return cmd

    def prepare_data(self, model_config):
        m_type = model_config['type']
        tool_name = m_type.replace('external_', '')
        params = model_config['params']
        print(f"\n[External] Executing PREPARE for {tool_name}...")
        cmd = self._build_base_cmd(tool_name) + ["prepare"]
        cmd.append("--dropdata")
        if tool_name == 'tpcc':
            cmd.extend(["--warehouses", str(params.get('warehouses', 1))])
        elif tool_name == 'tpch':
            cmd.extend(["--sf", str(params.get('scale_factor', 1))])
        if 'extra_args' in params:
            cmd.extend(params['extra_args'].split())
        self._run_subprocess(cmd)

    def run_workload(self, model_config):
        m_type = model_config['type']
        tool_name = m_type.replace('external_', '')
        base_params = model_config['params']
        inc_params = model_config.get('incremental', {})
        print(f"\n[External] Executing RUN (Incremental) for {tool_name}...")
        cmd = self._build_base_cmd(tool_name) + ["run"]
        if tool_name == 'tpcc':
            cmd.extend(["--warehouses", str(base_params.get('warehouses', 1))])
            cmd.extend(["--time", str(inc_params.get('time', '1m'))])
            cmd.extend(["--threads", str(inc_params.get('threads', 4))])
        elif tool_name == 'tpch':
            cmd.extend(["--sf", str(base_params.get('scale_factor', 1))])
            if 'queries' in inc_params:
                q_str = ",".join(inc_params['queries'])
                cmd.extend(["--queries", q_str])
        self._run_subprocess(cmd)

    def _run_subprocess(self, cmd):
        print(f"  Command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print("  -> External command finished successfully.")
        except subprocess.CalledProcessError as e:
            print(f"  -> External command failed with exit code {e.returncode}")
        except FileNotFoundError:
            print("  -> Error: 'tiup' command not found.")


# ==========================================
# 2. 数据库管理器
# ==========================================
class DBManager:
    def __init__(self, db_config):
        self.cfg = db_config
        self.conn = None
        self._connect()

    def _connect(self):
        try:
            self.conn = pymysql.connect(
                host=self.cfg['host'],
                port=self.cfg['port'],
                user=self.cfg['user'],
                password=self.cfg.get('password', ''),
                database=None,
                charset=self.cfg['charset'],
                local_infile=True,
                autocommit=True
            )
            print(f">>> Connected to Database: {self.cfg['host']}:{self.cfg['port']}")
        except Exception as e:
            print(f"Error connecting to DB: {e}")
            sys.exit(1)

    def ensure_connection(self):
        """确保连接可用，如果断开或脏了则重连"""
        try:
            self.conn.ping(reconnect=True)
        except:
            print("    [DB] Reconnecting...")
            self._connect()

    def init_db(self):
        db_name = self.cfg['db_name']
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
                cursor.execute(f"USE {db_name}")
            self.conn.select_db(db_name)
            print(f">>> Selected Database: {db_name}")
        except Exception as e:
            print(f"Error initializing DB: {e}")

    def disable_auto_analyze(self):
        """关闭TiDB自动Analyze [NEW]"""
        print("    [DB] Disabling Global Auto Analyze...")
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SET GLOBAL tidb_enable_auto_analyze = OFF;")
        except Exception as e:
            print(f"    [Warning] Failed to disable auto analyze: {e}")

    def create_table(self, table_name, df_preview):
        cols = []
        for col_name, dtype in df_preview.dtypes.items():
            if 'int' in str(dtype):
                sql_type = "BIGINT"
            elif 'datetime' in str(dtype):
                sql_type = "DATETIME"
            else:
                sql_type = "VARCHAR(255)"
            cols.append(f"`{col_name}` {sql_type}")
        indexes = [f"KEY `idx_{col}` (`{col}`)" for col in df_preview.columns]
        id_c = "`id` bigint NOT NULL AUTO_INCREMENT"
        pk = "PRIMARY KEY (`id`)"
        ddl = f"""CREATE TABLE IF NOT EXISTS `{table_name}` (
            {id_c}, {", ".join(cols)}, {", ".join(indexes)}, {pk}
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;"""
        # print("ddl:",ddl)
        with self.conn.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")
            cursor.execute(ddl)
        print(f"    [DB] Table created: {table_name}")

    def load_data_infile(self, table_name, csv_path):
        abs_path = os.path.abspath(csv_path).replace('\\', '/')
        sql = (f"LOAD DATA LOCAL INFILE '{abs_path}' INTO TABLE `{table_name}` "
               f"FIELDS TERMINATED BY ',' ENCLOSED BY '\"' "
               f"LINES TERMINATED BY '\\n' IGNORE 1 LINES;")
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
            print(f"    [DB] Data loaded into {table_name}")
        except Exception as e:
            print(f"    [Error] Load Data failed: {e}")

    def get_single_table_health(self, table_name):
        """[新增] 获取单个表的健康度"""
        db_name = self.cfg['db_name']
        try:
            with self.conn.cursor() as cursor:
                # 精确查询特定表的健康度
                sql = f"SHOW STATS_HEALTHY WHERE Db_name = '{db_name}' AND Table_name = '{table_name}'"
                cursor.execute(sql)
                res = cursor.fetchone()
                if res and len(res) >= 4:
                    return int(res[3])  # Healthy 字段
        except Exception as e:
            pass  # 忽略错误，可能表还没统计信息
        return 0

    def analyze_table(self, table_name):
        """手动Analyze表，并等待健康度变为100"""
        print(f"    [DB] Executing Manual Analyze: ANALYZE TABLE `{table_name}` ALL COLUMNS ...")
        start_ts = time.time()
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(f"ANALYZE TABLE `{table_name}` ALL COLUMNS")

            # [新增] 循环检查健康度，直到 100 或超时
            print(f"    [DB] Waiting for stats to become healthy (100%)...")
            max_retries = 20
            for i in range(max_retries):
                health = self.get_single_table_health(table_name)
                if health == 100:
                    break
                time.sleep(1)  # 等待元数据刷新
                if i == max_retries - 1:
                    print(f"    [Warning] Stats health reached {health}%, timed out waiting for 100%.")

            duration = time.time() - start_ts
            print(f"    [DB] Analyze finished in {duration:.2f}s (Health: {self.get_single_table_health(table_name)}%)")

        except Exception as e:
            print(f"    [Error] Analyze failed: {e}")

    def execute_sql_file(self, sql_path):
        print(f"    [DB] Executing SQL script: {sql_path}")
        if not os.path.exists(sql_path): return
        with open(sql_path, 'r', encoding='utf-8') as f:
            statements = f.read().split(';')
        with self.conn.cursor() as cursor:
            db_name = self.cfg['db_name']
            cursor.execute(f"USE {db_name}")
            for sql in statements:
                if sql.strip():
                    try:
                        cursor.execute(sql)
                    except Exception as e:
                        print(f"      SQL Error: {e}")

    def execute_and_explain(self, query_file):
        if not os.path.exists(query_file): return []
        with open(query_file, 'r', encoding='utf-8') as f:
            queries = [q.strip() for q in f.read().split(';') if q.strip()]
        results = []
        with self.conn.cursor() as cursor:
            db_name = self.cfg['db_name']
            cursor.execute(f"USE {db_name}")
            for i, sql in enumerate(queries):
                if sql.startswith('--'): continue
                start = time.time()
                try:
                    cursor.execute(sql)
                    cursor.fetchall()
                    dur = (time.time() - start) * 1000
                    cursor.execute(f"EXPLAIN analyze {sql}")
                    expl = "\n".join([str(r) for r in cursor.fetchall()])

                    # 解析EXPLAIN ANALYZE结果，计算estimation error
                    est_error_value, est_error_ratio, risk_count = self.parse_explain_analyze(expl)

                    results.append({
                        "query_id": i + 1,
                        "query": sql,
                        "duration_ms": dur,
                        "explain": expl,
                        "estimation_error_value": est_error_value,
                        "estimation_error_ratio": est_error_ratio,
                        "risk_operators_count": risk_count
                    })
                except Exception as e:
                    print(f"      Q{i + 1} Error: {e}")
        return results

    def get_table_stats(self, table_name, columns):
        self.ensure_connection()
        stats = {}
        with self.conn.cursor() as cursor:
            db_name = self.cfg['db_name']
            cursor.execute(f"USE {db_name}")
            for col in columns:
                try:
                    # 使用反引号包裹列名，防止关键字冲突
                    sql = f"SELECT MIN(`{col}`), MAX(`{col}`) FROM `{table_name}`"
                    cursor.execute(sql)
                    res = cursor.fetchone()
                    if res:
                        stats[col] = {'min': res[0], 'max': res[1]}
                except Exception as e:
                    print(f"      [Warning] Failed to fetch stats for {table_name}.{col}: {e}")
        return stats

    def get_stats_healthy(self):
        """获取所有表的统计信息健康度"""
        self.ensure_connection()
        stats_healthy = {}

        with self.conn.cursor() as cursor:
            db_name = self.cfg['db_name']
            cursor.execute(f"USE {db_name}")

            try:
                # 执行show stats_healthy命令
                cursor.execute("SHOW STATS_HEALTHY")
                results = cursor.fetchall()

                for row in results:
                    if len(row) >= 4:
                        table_name = str(row[1])
                        try:
                            healthy_ratio = float(row[3]) / 100.0 if row[3] is not None else 1.0
                            stats_healthy[table_name] = healthy_ratio
                        except (ValueError, TypeError):
                            stats_healthy[table_name] = 1.0

            except Exception as e:
                print(f"Warning: Could not execute SHOW STATS_HEALTY: {e}")
                return {}

        return stats_healthy

    def parse_explain_analyze(self, explain_text):
        """解析 EXPLAIN ANALYZE 输出"""
        try:
            import ast
            import re
            operators = []

            for line in explain_text.split('\n'):
                line = line.strip()
                if not line.startswith('(') or not line.endswith(')'):
                    continue

                try:
                    tuple_data = ast.literal_eval(line)
                    if len(tuple_data) >= 3:
                        operator_name = str(tuple_data[0])
                        est_rows_str = str(tuple_data[1]).strip()
                        act_rows_str = str(tuple_data[2]).strip()

                        est_match = re.search(r'(\d+\.?\d*)', est_rows_str)
                        act_match = re.search(r'(\d+\.?\d*)', act_rows_str)

                        if est_match and act_match:
                            est_rows = max(float(est_match.group(1)),1)
                            act_rows = max(float(act_match.group(1)),1)
                            original_act_rows = act_rows

                            estimation_error_value = abs(est_rows - original_act_rows)
                            estimation_error_ratio = max(act_rows, est_rows) / min(act_rows, est_rows)

                            is_risk = (estimation_error_ratio >= 10) and (estimation_error_value >= 1000)

                            operators.append({
                                'name': operator_name,
                                'est_rows': est_rows,
                                'act_rows': act_rows,
                                'estimation_error_value': estimation_error_value,
                                'estimation_error_ratio': estimation_error_ratio,
                                'is_risk': is_risk
                            })
                except:
                    continue

            if not operators:
                return 0.0, 0.0, 0

            avg_error_value = sum(op['estimation_error_value'] for op in operators) / len(operators)
            avg_error_ratio = sum(op['estimation_error_ratio'] for op in operators) / len(operators)
            risk_count = sum(1 for op in operators if op['is_risk'])

            return avg_error_value, avg_error_ratio, risk_count
        except Exception as e:
            print(f"Warning: Error parsing EXPLAIN ANALYZE: {e}")
            return 0.0, 0.0, 0


def get_stats_healthy_for_model(model_name, stats_healthy_info):
    try:
        table_name = model_name
        if table_name in stats_healthy_info:
            return stats_healthy_info[table_name]
        else:
            return 1.0
    except Exception as e:
        print(f"Warning: Error getting stats healthy for model {model_name}: {e}")
        return 1.0


def calculate_modify_ratio(model_name, config_data):
    try:
        for model in config_data.get('models', []):
            if model.get('name') == model_name:
                params = model.get('params', {})
                incremental = model.get('incremental', {})

                base_rows = float(params.get('rows', 0))
                insert_rows = float(incremental.get('insert_rows', 0))
                update_ratio = float(incremental.get('update_ratio', 0))
                delete_ratio = float(incremental.get('delete_ratio', 0))

                if base_rows > 0:
                    modify_ratio = (insert_rows / base_rows) + update_ratio + delete_ratio
                    return modify_ratio
                else:
                    return 0.0
    except Exception as e:
        print(f"Warning: Error calculating modify_ratio for {model_name}: {e}")
        return 0.0
    return 0.0


def generate_html_report(df_report, html_filename, columns, config_data=None):
    try:
        import html
        from datetime import datetime
        import json

        if config_data is None:
            try:
                with open('config.json', 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config.json: {e}")
                config_data = {"models": []}

        total_queries = len(df_report)
        total_bad_case = len(
            df_report[(df_report['estimation_error_ratio'] >= 10) & (df_report['estimation_error_value'] >= 1000)])

        html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Query Execution Report - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; text-align: center; margin-bottom: 30px; }}
        .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 12px; }}
        th {{ background-color: #4CAF50; color: white; padding: 12px 8px; text-align: left; font-weight: bold; position: sticky; top: 0; z-index: 10; }}
        td {{ padding: 8px; border-bottom: 1px solid #ddd; vertical-align: top; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f0f0f0; }}
        .high-error {{ background-color: #ffebee !important; color: #c62828; font-weight: bold; }}
        .query-cell {{ max-width: 300px; word-wrap: break-word; font-family: 'Courier New', monospace; font-size: 11px; }}
        .explain-cell {{ max-width: 400px; word-wrap: break-word; font-family: 'Courier New', monospace; font-size: 10px; color: #666; white-space: pre-wrap; }}
        .numeric-cell {{ text-align: left; font-family: 'Courier New', monospace; }}
        .stats {{ display: flex; justify-content: space-around; margin-bottom: 20px; }}
        .stat-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; min-width: 120px; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .stat-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Query Execution Analysis Report</h1>
        <div class="summary">
            <strong>Report Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
            <strong>Total Queries:</strong> {len(df_report)}
        </div>

        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{total_queries}</div>
                <div class="stat-label">Total Queries</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{total_bad_case}</div>
                <div class="stat-label">Total Bad Case<br>(Risk Queries)</div>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Stats Healthy Ratio</th>
                    <th>Modify Ratio</th>
                    <th>Query Label</th>
                    <th>Estimation Error Ratio</th>
                    <th>Estimation Error Value</th>
                    <th>Query SQL</th>
                    <th>Duration (ms)</th>
                    <th>Explain Plan</th>
                </tr>
            </thead>
            <tbody>
'''

        for _, row in df_report.head(100).iterrows():
            error_ratio = float(row['estimation_error_ratio'])
            error_value = float(row['estimation_error_value'])
            is_risk_query = (error_ratio >= 10) and (error_value >= 1000)

            row_class = 'high-error' if is_risk_query else ''

            model = html.escape(str(row['Model']))
            stats_healthy = float(row.get('stats_healthy_ratio', 1.0))
            modify_ratio = calculate_modify_ratio(str(row['Model']), config_data)
            query_label = html.escape(str(row.get('query_label', '')))
            error_ratio_val = float(row['estimation_error_ratio'])
            error_value_display = float(row['estimation_error_value'])
            query_sql = html.escape(str(row['query']))
            duration = float(row['duration_ms'])
            explain = html.escape(str(row['explain']))

            html_content += f'''
                <tr class="{row_class}">
                    <td>{model}</td>
                    <td class="numeric-cell">{stats_healthy:.3f}</td>
                    <td class="numeric-cell">{modify_ratio:.3f}</td>
                    <td>{query_label}</td>
                    <td class="numeric-cell">{error_ratio_val:.2f}</td>
                    <td class="numeric-cell">{error_value_display:.2f}</td>
                    <td class="query-cell">{query_sql}</td>
                    <td class="numeric-cell">{duration:.3f}</td>
                    <td class="explain-cell">{explain}</td>
                </tr>
'''

        html_content += '''
            </tbody>
        </table>
    </div>
</body>
</html>
'''
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

    except Exception as e:
        print(f"Error generating HTML report: {e}")


class SqlGenerator:
    def __init__(self):
        self.s = []

    def _fmt(self, v):
        return "NULL" if pd.isna(v) else f"'{v}'" if isinstance(v, (str, datetime, pd.Timestamp)) else str(v)

    def log_delete_limit(self, table_name, limit_count):
        if limit_count > 0:
            self.s.append(f"DELETE FROM `{table_name}` LIMIT {limit_count};")

    def log_upd(self, tbl, col, vals, cnames, mat):
        for i, oid in enumerate(vals):
            sets = [f"{n}={self._fmt(mat[i][j])}" for j, n in enumerate(cnames)]
            self.s.append(f"UPDATE {tbl} SET {', '.join(sets)} WHERE {col} = {self._fmt(oid)};")

    def log_ins(self, tbl, df):
        cols = ", ".join(df.columns)
        for _, r in df.iterrows(): self.s.append(
            f"INSERT INTO {tbl} ({cols}) VALUES ({', '.join([self._fmt(x) for x in r])});")

    def save(self, fn):
        with open(fn, 'w', encoding='utf-8') as f: f.write("\n".join(self.s))


class QueryBuilder:
    def generate(self, model_config, table_name, output_file, current_stats=None):
        old_content = ""
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        old_content += f"\n\n-- ========================================================\n-- [ARCHIVED HISTORY] Generated before {timestamp}\n-- ========================================================\n"
                        for line in lines:
                            old_content += f"-- {line}"
            except Exception as e:
                print(f"Warning: Failed to read old query file: {e}")

        sqls = [f"-- Auto-generated for {table_name} at {datetime.now()}"]
        p = model_config['params']
        m_type = model_config['type']
        col_int, col_str, col_dt = f"{model_config['name']}_int", f"{model_config['name']}_varchar", f"{model_config['name']}_datetime"

        if current_stats and col_int in current_stats and current_stats[col_int]['max'] is not None:
            max_i, min_i = current_stats[col_int]['max'], current_stats[col_int]['min']
        else:
            min_i, max_i = p.get('int_range', [0, 100])

        sqls.append(f"SELECT /*+ IGNORE_INDEX({table_name} PRIMARY) */ 1 FROM {table_name} WHERE {col_int} = {max_i + 1000}")
        sqls.append(f"SELECT /*+ IGNORE_INDEX({table_name} PRIMARY) */ 1 FROM {table_name} WHERE {col_int} = {min_i + 1}")
        sqls.append(f"SELECT /*+ IGNORE_INDEX({table_name} PRIMARY) */ 1 FROM {table_name} WHERE {col_int} BETWEEN {min_i} AND {min_i + 50}")

        if m_type == 'holes' and 'int_hole_range' in p:
            h_start, h_end = p['int_hole_range']
            sqls.append(f"-- [Int] Holes Specific Queries")
            sqls.append(f"SELECT /*+ IGNORE_INDEX({table_name} PRIMARY) */ 1 FROM {table_name} WHERE {col_int} > {h_start} AND {col_int} < {h_end}")
            offset = max(int((h_end - h_start) * 0.1), 500)
            cross_start = max(min_i, h_start - offset)
            cross_end = h_start + offset
            sqls.append(f"SELECT /*+ IGNORE_INDEX({table_name} PRIMARY) */ 1 FROM {table_name} WHERE {col_int} > {cross_start} AND {col_int} < {cross_end}")

        v_conf = p.get('varchar_range', {})
        prefix = v_conf.get('prefix', 'user_')
        s_min, s_max = v_conf.get('suffix_range', [1, 1000])
        sqls.append(f"SELECT /*+ IGNORE_INDEX({table_name} PRIMARY) */ 1 FROM {table_name} WHERE {col_str} = '{prefix}{s_max + 1000}'")
        sqls.append(f"SELECT /*+ IGNORE_INDEX({table_name} PRIMARY) */ 1 FROM {table_name} WHERE {col_str} = '{prefix}{s_min + 1}'")
        sqls.append(f"SELECT /*+ IGNORE_INDEX({table_name} PRIMARY) */ 1 FROM {table_name} WHERE {col_str} BETWEEN '{prefix}{s_min}' AND '{prefix}{s_min + 50}'")

        d_range = p.get('date_range', ["2024-01-01", "2024-12-31"])
        if current_stats and col_dt in current_stats and current_stats[col_dt]['max'] is not None:
            real_max_val = str(current_stats[col_dt]['max'])
        else:
            real_max_val = d_range[1]

        dt_min_str = d_range[0]
        try:
            dt_min = pd.to_datetime(dt_min_str)
        except:
            dt_min = datetime.now()

        sqls.append(f"SELECT /*+ IGNORE_INDEX({table_name} PRIMARY) */ 1 FROM {table_name} WHERE {col_dt} > '{real_max_val}'")
        dt_eq = (dt_min + timedelta(days=1)).strftime("%Y-%m-%d")
        sqls.append(f"SELECT /*+ IGNORE_INDEX({table_name} PRIMARY) */ 1 FROM {table_name} WHERE {col_dt} = '{dt_eq}'")
        dt_range_end = (dt_min + timedelta(days=30)).strftime("%Y-%m-%d")
        dt_min_str_val = dt_min.strftime("%Y-%m-%d")
        sqls.append(f"SELECT /*+ IGNORE_INDEX({table_name} PRIMARY) */ 1 FROM {table_name} WHERE {col_dt} BETWEEN '{dt_min_str_val}' AND '{dt_range_end}'")

        if m_type == 'holes' and 'date_hole_range' in p:
            dh_start_str, dh_end_str = p['date_hole_range']
            sqls.append(f"-- [Datetime] Holes Specific Queries")
            sqls.append(f"SELECT /*+ IGNORE_INDEX({table_name} PRIMARY) */ 1 FROM {table_name} WHERE {col_dt} > '{dh_start_str}' AND {col_dt} < '{dh_end_str}'")
            try:
                dh_start = pd.to_datetime(dh_start_str)
                dh_end = pd.to_datetime(dh_end_str)
                gap_delta = dh_end - dh_start
                offset = max(gap_delta * 0.1, timedelta(days=1))
                cross_start = (dh_start - offset).strftime("%Y-%m-%d")
                cross_end = (dh_start + offset).strftime("%Y-%m-%d")
                sqls.append(f"SELECT /*+ IGNORE_INDEX({table_name} PRIMARY) */ 1 FROM {table_name} WHERE {col_dt} > '{cross_start}' AND {col_dt} < '{cross_end}'")
            except Exception as e:
                sqls.append(f"-- Error generating datetime crossing query: {e}")

        sqls.append(f"SELECT /*+ IGNORE_INDEX({table_name} PRIMARY) */ 1 FROM {table_name} WHERE {col_int} > {min_i} AND {col_str} LIKE '{prefix}%'")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(";\n".join(sqls))
            f.write(";\n")
            if old_content:
                f.write(old_content)


class DataGenerator:
    def _generate_date_pool(self, start_str, end_str, size):
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        delta = (end - start).days
        num_points = size if delta > 0 else 1
        return [start + timedelta(days=int(x)) for x in np.linspace(0, delta, num=num_points)]

    def _build_prob_dist(self, weights, size):
        sum_w = sum(weights)
        if sum_w > 1.0: weights = [w / sum_w for w in weights]
        remain_prob = 1.0 - sum(weights)
        remain_cnt = size - len(weights)
        if remain_cnt > 0:
            return weights + [remain_prob / remain_cnt] * remain_cnt
        else:
            sub = weights[:size]
            s = sum(sub)
            return [x / s for x in sub]

    def _dpool(self, s, e, sz):
        start, end = datetime.strptime(s, "%Y-%m-%d"), datetime.strptime(e, "%Y-%m-%d")
        delta = (end - start).days
        return [start + timedelta(days=x) for x in np.linspace(0, delta, num=sz if delta > 0 else 1)]

    def _prob(self, w, sz):
        if sum(w) > 1.0: w = [x / sum(w) for x in w]
        rem = 1.0 - sum(w)
        rem_cnt = sz - len(w)
        return w + [rem / rem_cnt] * rem_cnt if rem_cnt > 0 else [x / sum(w[:sz]) for x in w[:sz]]

    def generate(self, model_config):
        m, p = model_config['type'], model_config['params']
        rows = int(p.get('rows', 1000))
        df = pd.DataFrame()

        if 'int_range' in p:
            s, e = p['int_range']
            if m == 'skew':
                ndv = p.get('ndv', rows)
                pool = np.linspace(s, e, ndv, dtype=int)
                np.random.shuffle(pool)
                df['col_int'] = np.random.choice(pool, size=rows, p=self._prob(p.get('skew_weights', [0.8, 0.2]), ndv))
            else:
                df['col_int'] = np.random.choice(np.linspace(s, e, p.get('ndv', rows), dtype=int), size=rows)
        else:
            df['col_int'] = np.arange(1, rows + 1)

        if 'varchar_range' in p:
            v = p['varchar_range']
            if 'options' in v:
                df['col_varchar'] = np.random.choice(v['options'], size=rows)
            else:
                s, e = v.get('suffix_range', [1, rows])
                df['col_varchar'] = [f"{v.get('prefix', '')}{random.randint(s, e)}" for _ in range(rows)]
        else:
            df['col_varchar'] = [fake.word() for _ in range(rows)]

        if 'date_range' in p:
            d_s, d_e = p['date_range']
            pool_date = self._generate_date_pool(d_s, d_e, p.get('ndv', 100))
            df['col_datetime'] = np.random.choice(pool_date, size=rows)
        else:
            df['col_datetime'] = [datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)] * rows

        if m == 'holes':
            if 'int_hole_range' in p:
                h_s, h_e = p['int_hole_range']
                df = df[~((df['col_int'] >= h_s) & (df['col_int'] <= h_e))]

            if 'date_hole_range' in p:
                dh_s_str, dh_e_str = p['date_hole_range']
                dh_s = pd.to_datetime(dh_s_str)
                dh_e = pd.to_datetime(dh_e_str)
                df = df[~((df['col_datetime'] >= dh_s) & (df['col_datetime'] <= dh_e))]

        df = df.sample(frac=1).reset_index(drop=True)
        if 'col_datetime' in df.columns:
            df['col_datetime'] = df['col_datetime'].apply(lambda x: x.strftime("%Y-%m-%d") if not pd.isna(x) else x)

        return df


def rename_columns(df, name):
    return df.rename(
        columns={'col_int': f'{name}_int', 'col_datetime': f'{name}_datetime', 'col_varchar': f'{name}_varchar'})


class DataModifier:
    def __init__(self, sql_gen):
        self.sql_gen = sql_gen

    def apply(self, df, cfg, name):
        inc, id_col, cols = cfg.get('incremental', {}), f"{name}_int", df.columns

        # Insert
        cnt = inc.get('insert_rows', 0)
        if cnt > 0:
            tmp = copy.deepcopy(cfg);
            tmp['params']['rows'] = cnt
            for k, v in inc.items():
                if k not in CONTROL_KEYS: tmp['params'][k] = v
            df_new = rename_columns(DataGenerator().generate(tmp), name)
            self.sql_gen.log_ins(name, df_new)
            df = pd.concat([df, df_new], ignore_index=True)

        # Update
        if inc.get('update_ratio', 0) > 0 and not df.empty and id_col in df:
            vidx = df[df[id_col].notna()].index
            cnt = int(len(vidx) * inc['update_ratio'])
            if cnt > 0:
                idx = np.random.choice(vidx, cnt, replace=True)
                tmp = copy.deepcopy(cfg);
                tmp['params']['rows'] = cnt
                for k, v in inc.items():
                    if k not in CONTROL_KEYS: tmp['params'][k] = v
                df_up = rename_columns(DataGenerator().generate(tmp), name)
                self.sql_gen.log_upd(name, id_col, df.loc[idx, id_col].values, cols, df_up[cols].values)
                df.loc[idx, cols] = df_up[cols].values

        # Delete

        if inc.get('delete_ratio', 0) > 0 and not df.empty:
            current_total = len(df)
            del_cnt = int(current_total * inc['delete_ratio'])

            if del_cnt > 0:
                self.sql_gen.log_delete_limit(name, del_cnt)
                valid_idx = df.index
                if len(valid_idx) > del_cnt:
                    drop_idx = np.random.choice(valid_idx, del_cnt, replace=False)
                    df.loc[drop_idx] = np.nan
                else:
                    df[:] = np.nan
        return df


def parse_arguments():
    parser = argparse.ArgumentParser(description="EstiGen Auto V17")
    parser.add_argument('--all', action='store_true', help="Execute all steps: Base -> Query -> Inc -> Exec Query")
    parser.add_argument('--gen-base', action='store_true', help="Step 1: Generate & Load Base Data")
    parser.add_argument('--gen-inc', action='store_true', help="Step 2: Generate & Execute Incremental Data")
    parser.add_argument('--gen-query', action='store_true',
                        help="Step 3: Generate SQL Queries (based on current DB stats)")
    parser.add_argument('--exec-query', action='store_true', help="Step 4: Execute SQL Queries & Report")
    parser.add_argument('--sql-file', type=str, default='incremental_dml.sql', help="File for incremental DML")
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.all:
        args.gen_base = True
        args.gen_inc = True
        args.gen_query = True
        args.exec_query = True

    try:
        with open('db_config.json', 'r') as f:
            db_conf = json.load(f)
        with open('config.json', 'r') as f:
            models = json.load(f)['models']
    except Exception as e:
        return print(f"Config Error: {e}")

    db = DBManager(db_conf)
    ext = ExternalBenchRunner(db_conf)
    gen, qb = DataGenerator(), QueryBuilder()

    # ==================================================
    # 1. Base Data Generation
    # ==================================================
    if args.gen_base:
        print("\n=== [Step 1] Base Data Generation ===")
        db.init_db()

        # 1. 关闭 Auto Analyze
        db.disable_auto_analyze()

        for m in models:
            name = m['name']
            if m['type'] in EXTERNAL_MODELS:
                ext.prepare_data(m)
            else:
                print(f"Generating base data for {name}...")
                df = rename_columns(gen.generate(m), name)
                csv = f"dataset_{name}_base.csv"
                df.to_csv(csv, index=False, lineterminator='\n')
                db.create_table(name, df)
                db.load_data_infile(name, csv)

        for m in models:
            name = m['name']
            if m['type'] in EXTERNAL_MODELS:
                ext.prepare_data(m)
            else:
                # 2. 手动执行 Analyze
                db.analyze_table(name)

    # ==================================================
    # 2. Incremental Data Generation & Execution
    # ==================================================
    if args.gen_inc:
        print("\n=== [Step 2] Incremental Data Update ===")
        sql_log = SqlGenerator()
        mod = DataModifier(sql_log)

        for m in models:
            name = m['name']
            if m['type'] in EXTERNAL_MODELS:
                ext.run_workload(m)
            else:
                base_csv = f"dataset_{name}_base.csv"
                if os.path.exists(base_csv):
                    print(f"Applying changes to {name}...")
                    df = pd.read_csv(base_csv)
                    if f"{name}_datetime" in df: df[f"{name}_datetime"] = pd.to_datetime(df[f"{name}_datetime"])
                    mod.apply(df, m, name)

        sql_log.save(args.sql_file)
        if os.path.exists(args.sql_file) and os.path.getsize(args.sql_file) > 0:
            print(f"Executing incremental DMLs from {args.sql_file}...")
            db.execute_sql_file(args.sql_file)

    # ==================================================
    # 3. Generate Queries (Based on CURRENT DB State)
    # ==================================================
    if args.gen_query:
        print("\n=== [Step 3] Generate Queries (Adaptive) ===")
        for m in models:
            if m['type'] in TARGET_QUERY_MODELS:
                name = m['name']
                cols = [f"{name}_int", f"{name}_datetime"]
                stats = db.get_table_stats(name, cols)

                outfile = f"queries_{name}.sql"
                qb.generate(m, name, outfile, current_stats=stats)
                print(f"Generated {outfile} based on DB stats: {stats}")

    # ==================================================
    # 4. Execute Queries & Report
    # ==================================================
    if args.exec_query:
        print("\n=== [Step 4] Execute Queries & Report ===")

        print("Getting stats healthy information...")
        stats_healthy_info = db.get_stats_healthy()
        print(f"Stats healthy info: {stats_healthy_info}")

        report = []
        for m in models:
            if m['type'] in TARGET_QUERY_MODELS:
                name = m['name']
                qfile = f"queries_{name}.sql"
                print(f"Executing {qfile}...")
                stats = db.execute_and_explain(qfile)
                for s in stats: s['Model'] = name
                report.extend(stats)

        if report:
            import html
            df_report = pd.DataFrame(report)
            config_with_models = {'models': models}

            def calc_healthy_ratio_for_row(row):
                model_name = str(row['Model'])
                return get_stats_healthy_for_model(model_name, stats_healthy_info)

            df_report['stats_healthy_ratio'] = df_report.apply(calc_healthy_ratio_for_row, axis=1)

            def calc_ratio_for_row(row):
                model_name = str(row['Model'])
                ratio = calculate_modify_ratio(model_name, config_with_models)
                return ratio

            df_report['modify_ratio'] = df_report.apply(calc_ratio_for_row, axis=1)
            df_report['query_label'] = ''

            df_report = df_report.sort_values('estimation_error_ratio', ascending=False)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_name = f"report_execution_{ts}.csv"
            csv_columns = ['Model', 'stats_healthy_ratio', 'modify_ratio', 'query_label', 'estimation_error_ratio',
                           'estimation_error_value', 'query', 'duration_ms', 'explain', 'risk_operators_count']
            df_report[csv_columns].to_csv(csv_name, index=False)
            print(f"CSV Report saved to: {csv_name}")

            html_name = f"report_execution_{ts}.html"
            generate_html_report(df_report, html_name, csv_columns, config_with_models)
            print(f"HTML Report saved to: {html_name}")

            display_columns = ['Model', 'stats_healthy_ratio', 'estimation_error_ratio', 'estimation_error_value',
                               'query']
            print(f"\nTop 10 queries by estimation error ratio:")
            print(df_report[display_columns].head(10).to_string())
        else:
            print("No queries executed or no results found.")


if __name__ == "__main__":
    main()