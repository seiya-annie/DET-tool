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
        ddl = f"""CREATE TABLE IF NOT EXISTS `{table_name}` (
            {", ".join(cols)}, {", ".join(indexes)}
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;"""
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
                    cursor.execute(f"EXPLAIN {sql}")
                    expl = "\n".join([str(r) for r in cursor.fetchall()])
                    results.append({"query_id": i + 1, "query": sql, "duration_ms": dur, "explain": expl})
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
                        # [调试日志] 打印查到的值，确认是否查到了 14000
                        # print(f"      [Debug] Stats for {col}: Min={res[0]}, Max={res[1]}")
                except Exception as e:
                    # [修复点 2] 打印具体的错误信息，而不是忽略
                    print(f"      [Warning] Failed to fetch stats for {table_name}.{col}: {e}")
                    # 如果发生 Commands out of sync，这里会打印出来
        return stats


# ==========================================
# 3. 内部生成器 (SQL/Query/Data/Modifier)
# ==========================================
class SqlGenerator:
    def __init__(self):
        self.s = []

    def _fmt(self, v):
        return "NULL" if pd.isna(v) else f"'{v}'" if isinstance(v, (str, datetime, pd.Timestamp)) else str(v)

    def log_del(self, tbl, col, vals):
        for v in vals: self.s.append(f"DELETE FROM {tbl} WHERE {col} = {self._fmt(v)};")

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
        # 注释掉旧的sql
        old_content = ""
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        old_content += f"\n\n"
                        old_content += f"-- ========================================================\n"
                        old_content += f"-- [ARCHIVED HISTORY] Generated before {timestamp}\n"
                        old_content += f"-- ========================================================\n"
                        for line in lines:
                            # 每一行前加 "-- "，保留原有换行符
                            old_content += f"-- {line}"
            except Exception as e:
                print(f"Warning: Failed to read old query file: {e}")

        # 生成新的查询语句
        sqls = [f"-- Auto-generated for {table_name} at {datetime.now()}"]
        p = model_config['params']
        m_type = model_config['type']
        col_int, col_str, col_dt = f"{model_config['name']}_int", f"{model_config['name']}_varchar", f"{model_config['name']}_datetime"

        # 1. Int (Use real stats if available)
        if current_stats and col_int in current_stats and current_stats[col_int]['max'] is not None:
            max_i, min_i = current_stats[col_int]['max'], current_stats[col_int]['min']
        else:
            min_i, max_i = p.get('int_range', [0, 100])

        sqls.append(f"SELECT * FROM {table_name} WHERE {col_int} = {max_i + 1000}")  # Out-of-range
        sqls.append(f"SELECT * FROM {table_name} WHERE {col_int} = {min_i + 1}")  # EQ
        sqls.append(f"SELECT * FROM {table_name} WHERE {col_int} BETWEEN {min_i} AND {min_i + 50}")  # Range

        if m_type == 'holes' and 'int_hole_range' in p:
            h_start, h_end = p['int_hole_range']

            sqls.append(f"-- [Int] Holes Specific Queries")

            sqls.append(f"SELECT * FROM {table_name} WHERE {col_int} > {h_start} AND {col_int} < {h_end}")

            # 为了模拟跨越边界，我们取空洞起始点的前后一段范围
            # 例如：空洞是 [4000, 6000]，跨越查询构造为 [3500, 4500]
            # 这里的 offset 取空洞大小的 10% 或固定值 500
            offset = max(int((h_end - h_start) * 0.1), 500)
            cross_start = max(min_i, h_start - offset)
            cross_end = h_start + offset

            sqls.append(f"SELECT * FROM {table_name} WHERE {col_int} > {cross_start} AND {col_int} < {cross_end}")
        # ----------------------------------------------------
        # 2. Varchar (Updated Logic)
        # ----------------------------------------------------
        # 规则：
        # - 获取 suffix_range 的 min, max
        # - Out of range: prefix + (max + 1000)
        # - EQ: prefix + (min + 1)
        # - Range: prefix + min TO prefix + (min + 50)
        # ----------------------------------------------------
        v_conf = p.get('varchar_range', {})
        prefix = v_conf.get('prefix', 'user_')
        s_min, s_max = v_conf.get('suffix_range', [1, 1000])
        # Out of range
        sqls.append(f"SELECT * FROM {table_name} WHERE {col_str} = '{prefix}{s_max + 1000}'")
        # EQ
        sqls.append(f"SELECT * FROM {table_name} WHERE {col_str} = '{prefix}{s_min + 1}'")
        # Range
        sqls.append(f"SELECT * FROM {table_name} WHERE {col_str} BETWEEN '{prefix}{s_min}' AND '{prefix}{s_min + 50}'")

        # ----------------------------------------------------
        # 3. Datetime (Updated Logic)
        # ----------------------------------------------------
        # 规则：
        # - 获取 date_range 的 min, max (或实时 stats)
        # - Out of range: > max
        # - EQ: = min + 1 day
        # - Range: min TO min + 30 days
        # ----------------------------------------------------
        d_range = p.get('date_range', ["2024-01-01", "2024-12-31"])

        # 获取真实 Max (用于 Out of Range)
        if current_stats and col_dt in current_stats and current_stats[col_dt]['max'] is not None:
            real_max_val = str(current_stats[col_dt]['max'])
        else:
            real_max_val = d_range[1]

        # 获取 Min (用于 EQ 和 Range)
        dt_min_str = d_range[0]
        try:
            dt_min = pd.to_datetime(dt_min_str)
        except:
            dt_min = datetime.now()

        # Out of range
        sqls.append(f"SELECT * FROM {table_name} WHERE {col_dt} > '{real_max_val}'")

        # EQ (min + 1 day)
        dt_eq = (dt_min + timedelta(days=1)).strftime("%Y-%m-%d")
        sqls.append(f"SELECT * FROM {table_name} WHERE {col_dt} = '{dt_eq}'")

        # Range (min TO min + 30 days)
        dt_range_end = (dt_min + timedelta(days=30)).strftime("%Y-%m-%d")
        dt_min_str = dt_min.strftime("%Y-%m-%d")

        sqls.append(f"SELECT * FROM {table_name} WHERE {col_dt} BETWEEN '{dt_min_str}' AND '{dt_range_end}'")
        # Hole
        if m_type == 'holes' and 'date_hole_range' in p:
            dh_start_str, dh_end_str = p['date_hole_range']

            sqls.append(f"-- [Datetime] Holes Specific Queries")

            # 1. 空洞区查询 (Inside the Gap)
            sqls.append(f"SELECT * FROM {table_name} WHERE {col_dt} > '{dh_start_str}' AND {col_dt} < '{dh_end_str}'")

            # 2. 跨空洞区查询 (Crossing the Gap Boundary)
            try:
                # 计算空洞跨度
                dh_start = pd.to_datetime(dh_start_str)
                dh_end = pd.to_datetime(dh_end_str)
                gap_delta = dh_end - dh_start

                # 定义 offset: 取空洞时间的 10% 或 至少 1 天
                offset = max(gap_delta * 0.1, timedelta(days=1))

                cross_start = (dh_start - offset).strftime("%Y-%m-%d")
                cross_end = (dh_start + offset).strftime("%Y-%m-%d")

                sqls.append(f"SELECT * FROM {table_name} WHERE {col_dt} > '{cross_start}' AND {col_dt} < '{cross_end}'")
            except Exception as e:
                sqls.append(f"-- Error generating datetime crossing query: {e}")

        # 4. Complex
        sqls.append(f"SELECT * FROM {table_name} WHERE {col_int} > {min_i} AND {col_str} LIKE '{prefix}%'")  # CNF

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(";\n".join(sqls))
            f.write(";\n")  # 确保最后一行有分号结束

            # 再追加注释掉的旧内容
            if old_content:
                f.write(old_content)


class DataGenerator:
    def _generate_date_pool(self, start_str, end_str, size):
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        delta = (end - start).days
        # Fix numpy linspace argument
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
                # 转换字符串为 datetime 对象进行比较
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
        if inc.get('delete_ratio', 0) > 0 and not df.empty and id_col in df:
            vidx = df[df[id_col].notna()].index
            cnt = int(len(vidx) * inc['delete_ratio'])
            if cnt > 0:
                idx = np.random.choice(vidx, cnt, replace=False)
                self.sql_gen.log_del(name, id_col, df.loc[idx, id_col].values)
                df.loc[idx] = np.nan
        return df


# ==========================================
# Main Argument Parsing & Flow
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="EstiGen Auto V17")
    parser.add_argument('--all', action='store_true', help="Execute all steps: Base -> Query -> Inc -> Exec Query")

    # Step Flags
    parser.add_argument('--gen-base', action='store_true', help="Step 1: Generate & Load Base Data")
    parser.add_argument('--gen-inc', action='store_true', help="Step 2: Generate & Execute Incremental Data")
    parser.add_argument('--gen-query', action='store_true',
                        help="Step 3: Generate SQL Queries (based on current DB stats)")
    parser.add_argument('--exec-query', action='store_true', help="Step 4: Execute SQL Queries & Report")

    parser.add_argument('--sql-file', type=str, default='incremental_dml.sql', help="File for incremental DML")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # 如果指定 --all，则开启所有步骤的 Flag
    if args.all:
        args.gen_base = True
        args.gen_inc = True
        args.gen_query = True  # 注意：逻辑上会在不同阶段调用
        args.exec_query = True

        # 加载配置
    try:
        with open('db_config.json', 'r') as f:
            db_conf = json.load(f)
        with open('config.json', 'r') as f:
            models = json.load(f)['models']
    except Exception as e:
        return print(f"Config Error: {e}")

    db = DBManager(db_conf);
    ext = ExternalBenchRunner(db_conf)
    gen, qb = DataGenerator(), QueryBuilder()

    # ==================================================
    # 1. Base Data Generation
    # ==================================================
    if args.gen_base:
        print("\n=== [Step 1] Base Data Generation ===")
        db.init_db()
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
                    # Simple type inference for modification logic
                    if f"{name}_datetime" in df: df[f"{name}_datetime"] = pd.to_datetime(df[f"{name}_datetime"])
                    mod.apply(df, m, name)

        sql_log.save(args.sql_file)
        if os.path.exists(args.sql_file) and os.path.getsize(args.sql_file) > 0:
            print(f"Executing incremental DMLs from {args.sql_file}...")
            db.execute_sql_file(args.sql_file)

    # ==================================================
    # 3. Generate Queries (Based on CURRENT DB State)
    # ==================================================
    # 这一步通常需要在 Data Ready 之后执行
    if args.gen_query:
        print("\n=== [Step 3] Generate Queries (Adaptive) ===")
        for m in models:
            if m['type'] in TARGET_QUERY_MODELS:
                name = m['name']
                # 获取实时统计信息，确保 Out-of-range 等查询有效
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
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_name = f"report_execution_{ts}.csv"
            pd.DataFrame(report).to_csv(csv_name, index=False)
            print(f"Report saved to: {csv_name}")
            print(f"Print head 10 sqls")
            # 简单打印
            print(pd.DataFrame(report)[['Model', 'duration_ms', 'query']].head(10).to_string())
        else:
            print("No queries executed or no results found.")


if __name__ == "__main__":
    main()