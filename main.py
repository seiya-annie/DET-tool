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
EXTERNAL_MODELS = ['tpcc', 'tpch', 'external_tpcc', 'external_tpch']
TARGET_QUERY_MODELS = ['skew', 'holes', 'low_card']
CONTROL_KEYS = ['insert_rows', 'update_ratio', 'delete_ratio']


# ==========================================
# 1. 外部工具运行器 (External Bench Runner)
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
            print("  -> Error: 'tiup' command not found. Please ensure TiUP is installed.")


# ==========================================
# 2. 数据库管理器 (DB Manager)
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
        ddl = f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            {", ".join(cols)},
            {", ".join(indexes)}
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
        """
        with self.conn.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")
            cursor.execute(ddl)
        print(f"    [DB] Table created: {table_name}")

    def load_data_infile(self, table_name, csv_path):
        abs_path = os.path.abspath(csv_path).replace('\\', '/')
        sql = (
            f"LOAD DATA LOCAL INFILE '{abs_path}' "
            f"INTO TABLE `{table_name}` "
            f"FIELDS TERMINATED BY ',' "
            f"ENCLOSED BY '\"' "
            f"LINES TERMINATED BY '\\n' "
            f"IGNORE 1 LINES;"
        )
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
            print(f"    [DB] Data loaded into {table_name}")
        except Exception as e:
            print(f"    [Error] Load Data failed: {e}")

    def execute_sql_file(self, sql_path):
        print(f"    [DB] Executing SQL script: {sql_path}")
        if not os.path.exists(sql_path):
            print("      File not found, skipping.")
            return
        with open(sql_path, 'r', encoding='utf-8') as f:
            statements = f.read().split(';')
        with self.conn.cursor() as cursor:
            for sql in statements:
                if sql.strip():
                    try:
                        cursor.execute(sql)
                    except Exception as e:
                        print(f"      SQL Error: {e} | SQL: {sql[:50]}...")

    def execute_and_explain(self, query_file):
        if not os.path.exists(query_file): return []
        with open(query_file, 'r', encoding='utf-8') as f:
            content = f.read()
            queries = [q.strip() for q in content.split(';') if q.strip()]
        results = []
        with self.conn.cursor() as cursor:
            for i, sql in enumerate(queries):
                if sql.startswith('--'): continue
                start_ts = time.time()
                try:
                    cursor.execute(sql)
                    cursor.fetchall()
                    duration = (time.time() - start_ts) * 1000
                    cursor.execute(f"EXPLAIN {sql}")
                    explain_rows = cursor.fetchall()
                    explain_str = "\n".join([str(row) for row in explain_rows]) if explain_rows else "No Plan"
                    results.append({
                        "query_id": i + 1,
                        "query": sql,
                        "duration_ms": duration,
                        "explain": explain_str
                    })
                except Exception as e:
                    print(f"      Q{i + 1} Error: {e}")
        return results

    def get_table_stats(self, table_name, columns):
        """
        [NEW] 实时查询数据库获取当前数据的 Min/Max 值
        返回: {'col_name': {'min': val, 'max': val}, ...}
        """
        stats = {}
        with self.conn.cursor() as cursor:
            for col in columns:
                try:
                    # 使用聚合查询获取真实范围
                    sql = f"SELECT MIN({col}), MAX({col}) FROM `{table_name}`"
                    cursor.execute(sql)
                    res = cursor.fetchone()
                    if res:
                        stats[col] = {'min': res[0], 'max': res[1]}
                except Exception as e:
                    print(f"      Warning: Could not fetch stats for {col}: {e}")
        return stats


# ==========================================
# 3. 内部数据生成器 (Internal Generators)
# ==========================================
class SqlGenerator:
    def __init__(self):
        self.statements = []

    def _format_val(self, val):
        if pd.isna(val): return "NULL"
        if isinstance(val, (str, datetime, pd.Timestamp)): return f"'{str(val)}'"
        return str(val)

    def log_delete(self, table_name, id_col, id_values):
        for val in id_values:
            self.statements.append(f"DELETE FROM {table_name} WHERE {id_col} = {self._format_val(val)};")

    def log_update(self, table_name, id_col, id_values, col_names, new_values_matrix):
        for i, old_id in enumerate(id_values):
            sets = [f"{col} = {self._format_val(new_values_matrix[i][j])}" for j, col in enumerate(col_names)]
            self.statements.append(
                f"UPDATE {table_name} SET {', '.join(sets)} WHERE {id_col} = {self._format_val(old_id)};")

    def log_insert(self, table_name, df_new):
        cols = ", ".join(df_new.columns)
        for _, row in df_new.iterrows():
            vals = ", ".join([self._format_val(x) for x in row])
            self.statements.append(f"INSERT INTO {table_name} ({cols}) VALUES ({vals});")

    def save_to_file(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.statements))


class QueryBuilder:
    def generate(self, model_config, table_name, output_file, current_stats=None):
        """
        自动构造测试查询语句
        current_stats: 字典, 包含从数据库查询到的实时 {'col': {'min':x, 'max':y}}
                       如果存在，优先使用实时数据构造 Out-of-range 查询
        """
        sqls = []
        params = model_config['params']
        sqls.append(f"-- Auto-generated queries for {table_name}")
        sqls.append(f"-- Timestamp: {datetime.now()}")

        # 1. Int Column
        col_int = f"{model_config['name']}_int"
        # 优先使用实时统计的 Max 值，否则使用配置值
        if current_stats and col_int in current_stats and current_stats[col_int]['max'] is not None:
            max_i = current_stats[col_int]['max']
            min_i = current_stats[col_int]['min']
        else:
            min_i, max_i = params.get('int_range', [0, 100])

        # Query 1: Out of range (Max + 1000)
        sqls.append(f"SELECT * FROM {table_name} WHERE {col_int} = {max_i + 1000}")
        # Query 2: EQ (Min + 1)
        sqls.append(f"SELECT * FROM {table_name} WHERE {col_int} = {min_i + 1}")
        # Query 3: Range (Min to Min+50)
        sqls.append(f"SELECT * FROM {table_name} WHERE {col_int} BETWEEN {min_i} AND {min_i + 50}")

        # 2. Varchar Column
        col_str = f"{model_config['name']}_varchar"
        prefix = params.get('varchar_range', {}).get('prefix', 'user_')
        # Query 4: EQ String
        sqls.append(f"SELECT * FROM {table_name} WHERE {col_str} = '{prefix}1'")

        # 3. Datetime Column
        col_dt = f"{model_config['name']}_datetime"
        if current_stats and col_dt in current_stats and current_stats[col_dt]['max'] is not None:
            # 使用真实最大时间
            d_start = current_stats[col_dt]['max']
        else:
            d_start = params.get('date_range', ["2024-01-01"])[0]

        # Query 5: Range Date
        sqls.append(f"SELECT * FROM {table_name} WHERE {col_dt} > '{d_start}'")

        # 4. Complex
        # Query 6: CNF
        sqls.append(f"SELECT * FROM {table_name} WHERE {col_int} > {min_i} AND {col_str} LIKE '{prefix}%'")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(";\n".join(sqls))


class DataGenerator:
    def _generate_date_pool(self, start_str, end_str, size):
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        delta = (end - start).days
        if delta <= 0: return [start] * size
        return [start + timedelta(days=x) for x in np.linspace(0, delta, size)]

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

    def generate(self, model_config):
        m_type = model_config['type']
        params = model_config['params']
        rows = int(params.get('rows', 1000))
        df = pd.DataFrame()

        if 'int_range' in params:
            i_start, i_end = params['int_range']
            if m_type == 'skew':
                ndv = params.get('ndv', rows)
                weights = params.get('skew_weights', [0.8, 0.2])
                pool_int = np.linspace(i_start, i_end, ndv, dtype=int)
                np.random.shuffle(pool_int)
                prob_dist = self._build_prob_dist(weights, ndv)
                df['col_int'] = np.random.choice(pool_int, size=rows, p=prob_dist)
            else:
                pool_int = np.linspace(i_start, i_end, params.get('ndv', rows), dtype=int)
                df['col_int'] = np.random.choice(pool_int, size=rows)
        else:
            df['col_int'] = np.arange(1, rows + 1)

        if 'varchar_range' in params:
            v_conf = params['varchar_range']
            if 'options' in v_conf:
                pool_str = v_conf['options']
                df['col_varchar'] = np.random.choice(pool_str, size=rows)
            else:
                prefix = v_conf.get('prefix', '')
                start, end = v_conf.get('suffix_range', [1, rows])
                df['col_varchar'] = [f"{prefix}{random.randint(start, end)}" for _ in range(rows)]
        else:
            df['col_varchar'] = [fake.word() for _ in range(rows)]

        if 'date_range' in params:
            d_s, d_e = params['date_range']
            pool_date = self._generate_date_pool(d_s, d_e, params.get('ndv', 100))
            df['col_datetime'] = np.random.choice(pool_date, size=rows)
        else:
            df['col_datetime'] = [datetime.now()] * rows

        if m_type == 'holes' and 'int_hole_range' in params:
            h_s, h_e = params['int_hole_range']
            df = df[~((df['col_int'] >= h_s) & (df['col_int'] <= h_e))]

        return df.sample(frac=1).reset_index(drop=True)


def rename_columns(df, model_name):
    return df.rename(columns={
        'col_int': f'{model_name}_int',
        'col_datetime': f'{model_name}_datetime',
        'col_varchar': f'{model_name}_varchar'
    })


class DataModifier:
    def __init__(self, sql_generator):
        self.sql_gen = sql_generator

    def apply_changes(self, df, model_config, table_name):
        inc_params = model_config.get('incremental', {})
        id_col = f"{model_config['name']}_int"
        cols = df.columns

        # 1. Insert
        cnt = inc_params.get('insert_rows', 0)
        if cnt > 0:
            temp_cfg = copy.deepcopy(model_config)
            temp_cfg['params']['rows'] = cnt
            for k, v in inc_params.items():
                if k not in CONTROL_KEYS: temp_cfg['params'][k] = v
            gen = DataGenerator()
            df_new = rename_columns(gen.generate(temp_cfg), model_config['name'])
            self.sql_gen.log_insert(table_name, df_new)
            df = pd.concat([df, df_new], ignore_index=True)

        # 2. Update
        if inc_params.get('update_ratio', 0) > 0 and not df.empty:
            if id_col in df.columns:
                valid_idx = df[df[id_col].notna()].index
                upd_cnt = int(len(valid_idx) * inc_params['update_ratio'])
                if upd_cnt > 0:
                    idx = np.random.choice(valid_idx, upd_cnt, replace=True)
                    gen = DataGenerator()
                    temp_cfg = copy.deepcopy(model_config)
                    temp_cfg['params']['rows'] = upd_cnt
                    for k, v in inc_params.items():
                        if k not in CONTROL_KEYS: temp_cfg['params'][k] = v
                    df_upd = rename_columns(gen.generate(temp_cfg), model_config['name'])
                    self.sql_gen.log_update(table_name, id_col, df.loc[idx, id_col].values, cols, df_upd[cols].values)
                    df.loc[idx, cols] = df_upd[cols].values

        # 3. Delete
        if inc_params.get('delete_ratio', 0) > 0 and not df.empty:
            if id_col in df.columns:
                valid_idx = df[df[id_col].notna()].index
                del_cnt = int(len(valid_idx) * inc_params['delete_ratio'])
                if del_cnt > 0:
                    idx = np.random.choice(valid_idx, del_cnt, replace=False)
                    self.sql_gen.log_delete(table_name, id_col, df.loc[idx, id_col].values)
                    df.loc[idx] = np.nan

        return df


# ==========================================
# Main Execution
# ==========================================
def main():
    try:
        with open('db_config.json', 'r', encoding='utf-8') as f:
            db_config = json.load(f)
        with open('config.json', 'r', encoding='utf-8') as f:
            full_config = json.load(f)
            model_config_list = full_config['models']
    except Exception as e:
        print(f"Error loading configs: {e}")
        return

    db = DBManager(db_config)
    db.init_db()

    ext_runner = ExternalBenchRunner(db_config)
    gen = DataGenerator()
    q_builder = QueryBuilder()

    # === Step 1: Base Data Generation ===
    print("\n" + "=" * 50)
    print(" STEP 1: Base Data Generation")
    print("=" * 50)

    for model in model_config_list:
        name = model['name']
        m_type = model['type']

        if m_type in EXTERNAL_MODELS:
            ext_runner.prepare_data(model)
        else:
            print(f"Generating internal data for {name}...")
            df = gen.generate(model)
            df = rename_columns(df, name)

            csv_file = f"dataset_{name}_base.csv"
            df.to_csv(csv_file, index=False, lineterminator='\n')

            db.create_table(name, df)
            db.load_data_infile(name, csv_file)

    # === Step 2: Query Generation & Execution (Base Stats) ===
    print("\n" + "=" * 50)
    print(" STEP 2: Initial Query Execution")
    print("=" * 50)

    query_files_base = {}
    baseline_stats = {}
    all_baseline_data = []

    for model in model_config_list:
        if model['type'] in INTERNAL_MODELS:
            name = model['name']

            # [NEW] 获取实时统计信息，用于生成更准确的初始查询
            print(f"  Fetching stats for {name}...")
            cols_to_stat = [f"{name}_int", f"{name}_datetime"]
            curr_stats = db.get_table_stats(name, cols_to_stat)

            q_file = f"queries_{name}_step2.sql"
            q_builder.generate(model, name, q_file, current_stats=curr_stats)
            query_files_base[name] = q_file

            print(f"  Executing queries for {name}...")
            stats = db.execute_and_explain(q_file)
            for s in stats: s['Model'] = name
            baseline_stats[name] = stats
            all_baseline_data.extend(stats)

    if all_baseline_data:
        pd.DataFrame(all_baseline_data).to_csv("report_step2_baseline.csv", index=False)
        print("  -> Saved baseline report to report_step2_baseline.csv")

    # === Step 3: Incremental Changes ===
    print("\n" + "=" * 50)
    print(" STEP 3: Incremental Data & Execution")
    print("=" * 50)

    sql_gen = SqlGenerator()
    modifier = DataModifier(sql_gen)

    for model in model_config_list:
        name = model['name']
        m_type = model['type']

        if m_type in EXTERNAL_MODELS:
            ext_runner.run_workload(model)
        else:
            base_csv = f"dataset_{name}_base.csv"
            if os.path.exists(base_csv):
                print(f"Calculating changes for {name}...")
                df_curr = pd.read_csv(base_csv)
                for col in df_curr.columns:
                    if 'datetime' in col: df_curr[col] = pd.to_datetime(df_curr[col], errors='coerce')

                modifier.apply_changes(df_curr, model, name)
            else:
                print(f"Warning: Base file for {name} not found.")

    inc_sql_file = "incremental_dml.sql"
    sql_gen.save_to_file(inc_sql_file)
    if os.path.exists(inc_sql_file) and os.path.getsize(inc_sql_file) > 0:
        db.execute_sql_file(inc_sql_file)
    else:
        print("No internal incremental SQL generated.")

    # === Step 4: Regression Testing (RE-GENERATE & Re-execute Queries) ===
    print("\n" + "=" * 50)
    print(" STEP 4: Regression Testing (Re-gen & Re-execute)")
    print("=" * 50)

    comparison_report = []
    all_regression_data = []

    for model in model_config_list:
        if model['type'] in INTERNAL_MODELS:
            name = model['name']

            # [NEW] 1. 获取增量更新后的实时统计信息
            print(f"  Fetching NEW stats for {name}...")
            cols_to_stat = [f"{name}_int", f"{name}_datetime"]
            new_stats = db.get_table_stats(name, cols_to_stat)

            # [NEW] 2. 重新生成查询 (基于新数据分布，例如 Max 值变大)
            q_file_step4 = f"queries_{name}_step4.sql"
            q_builder.generate(model, name, q_file_step4, current_stats=new_stats)
            print(f"  Generated updated queries: {q_file_step4}")

            # 3. 执行新查询
            new_exec_stats = db.execute_and_explain(q_file_step4)
            for s in new_exec_stats: s['Model'] = name
            all_regression_data.extend(new_exec_stats)

            # 4. 对比 (对比的是 Scenario ID，即第i个查询 vs 第i个查询)
            old_list = baseline_stats.get(name, [])
            for i, new_res in enumerate(new_exec_stats):
                if i < len(old_list):
                    old_res = old_list[i]
                    diff = new_res['duration_ms'] - old_res['duration_ms']
                    plan_changed = (new_res['explain'] != old_res['explain'])

                    comparison_report.append({
                        "Model": name,
                        "Query_ID": new_res['query_id'],
                        "Step2_SQL": old_res['query'][:30] + "...",
                        "Step4_SQL": new_res['query'][:30] + "...",
                        "Old_Time_ms": f"{old_res['duration_ms']:.2f}",
                        "New_Time_ms": f"{new_res['duration_ms']:.2f}",
                        "Diff_ms": f"{diff:+.2f}",
                        "Plan_Changed": "YES" if plan_changed else "NO"
                    })

    if all_regression_data:
        pd.DataFrame(all_regression_data).to_csv("report_step4_regression.csv", index=False)
        print("  -> Saved regression report to report_step4_regression.csv")

    print("\n=== Final Comparison Report ===")
    if comparison_report:
        df_comp = pd.DataFrame(comparison_report)
        print(df_comp[['Model', 'Query_ID', 'Step4_SQL', 'Diff_ms', 'Plan_Changed']].to_string())
        df_comp.to_csv("report_final_comparison.csv", index=False)
        print("  -> Saved comparison report to report_final_comparison.csv")
    else:
        print("No queries tracked for comparison.")


if __name__ == "__main__":
    main()