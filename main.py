import pandas as pd
import numpy as np
from faker import Faker
import json
import random
import argparse
import os
import sys
import copy
from datetime import datetime, timedelta

fake = Faker()

# 控制参数Key，用于区分是"数量控制"还是"模型参数"
CONTROL_KEYS = ['insert_rows', 'update_ratio', 'delete_ratio']


class SqlGenerator:
    def __init__(self):
        self.statements = []

    def _format_val(self, val):
        if pd.isna(val):
            return "NULL"
        if isinstance(val, (str, datetime, pd.Timestamp)):
            return f"'{str(val)}'"
        return str(val)

    def log_delete(self, table_name, id_col, id_values):
        for val in id_values:
            sql = f"DELETE FROM {table_name} WHERE {id_col} = {self._format_val(val)};"
            self.statements.append(sql)

    def log_update(self, table_name, id_col, id_values, col_names, new_values_matrix):
        for i, old_id in enumerate(id_values):
            set_clauses = []
            for j, col in enumerate(col_names):
                val = new_values_matrix[i][j]
                set_clauses.append(f"{col} = {self._format_val(val)}")

            set_str = ", ".join(set_clauses)
            sql = f"UPDATE {table_name} SET {set_str} WHERE {id_col} = {self._format_val(old_id)};"
            self.statements.append(sql)

    def log_insert(self, table_name, df_new):
        cols = ", ".join(df_new.columns)
        for _, row in df_new.iterrows():
            vals = ", ".join([self._format_val(x) for x in row])
            sql = f"INSERT INTO {table_name} ({cols}) VALUES ({vals});"
            self.statements.append(sql)

    def save_to_file(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.statements))
            f.write("\n")


class DataGenerator:
    def _generate_date_pool(self, start_str, end_str, size):
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        delta = (end - start).days
        return [start + timedelta(days=x) for x in np.linspace(0, delta, size)]

    def generate(self, model_config):
        m_type = model_config['type']
        params = model_config['params']
        rows = int(params.get('rows', 1000))  # 默认值兜底

        df = pd.DataFrame()

        # === 1. Skew ===
        if m_type == 'skew':
            ndv = params.get('ndv', 1000)
            weights = params.get('skew_weights', [0.99, 0.005])

            sum_weights = sum(weights)
            if sum_weights > 1.0:
                weights = [w / sum_weights for w in weights]
                sum_weights = 1.0

            remaining_prob = 1.0 - sum_weights
            remaining_count = ndv - len(weights)

            if remaining_count > 0:
                probs = weights + [remaining_prob / remaining_count] * remaining_count
            else:
                probs = weights[:ndv]
                total = sum(probs)
                probs = [p / total for p in probs]

            i_start, i_end = params['int_range']
            pool_int = np.linspace(i_start, i_end, ndv, dtype=int)
            np.random.shuffle(pool_int)
            df['col_int'] = np.random.choice(pool_int, size=rows, p=probs)

            v_conf = params['varchar_range']
            v_start, v_end = v_conf['suffix_range']
            suffix_pool = np.linspace(v_start, v_end, ndv, dtype=int)
            pool_str = np.array([f"{v_conf['prefix']}{x}" for x in suffix_pool])
            np.random.shuffle(pool_str)
            df['col_varchar'] = np.random.choice(pool_str, size=rows, p=probs)

            d_start, d_end = params['date_range']
            pool_date = np.array(self._generate_date_pool(d_start, d_end, ndv))
            np.random.shuffle(pool_date)
            df['col_datetime'] = np.random.choice(pool_date, size=rows, p=probs)

        # === 2. Holes ===
        elif m_type == 'holes':
            total_gen_size = int(params['int_range'][1])
            if total_gen_size < rows: total_gen_size = rows + 5000

            full_int = np.arange(1, total_gen_size + 1)
            d_start_str, d_end_str = params['date_range']
            full_date = np.array(self._generate_date_pool(d_start_str, d_end_str, total_gen_size))
            v_conf = params['varchar_range']
            v_suffix = np.linspace(v_conf['suffix_range'][0], v_conf['suffix_range'][1], total_gen_size, dtype=int)
            full_varchar = np.array([f"{v_conf['prefix']}{x}" for x in v_suffix])

            df_temp = pd.DataFrame({'col_int': full_int, 'col_datetime': full_date, 'col_varchar': full_varchar})

            if 'int_hole_range' in params:
                h_start, h_end = params['int_hole_range']
                mask = (df_temp['col_int'] >= h_start) & (df_temp['col_int'] <= h_end)
                df_temp = df_temp[~mask]
            if 'date_hole_range' in params:
                dh_start = datetime.strptime(params['date_hole_range'][0], "%Y-%m-%d")
                dh_end = datetime.strptime(params['date_hole_range'][1], "%Y-%m-%d")
                mask = (df_temp['col_datetime'] >= dh_start) & (df_temp['col_datetime'] <= dh_end)
                df_temp = df_temp[~mask]
            if 'varchar_hole_range' in params:
                vh_start = params['varchar_hole_range']['start']
                vh_end = params['varchar_hole_range']['end']
                mask = (df_temp['col_varchar'] >= vh_start) & (df_temp['col_varchar'] <= vh_end)
                df_temp = df_temp[~mask]

            if len(df_temp) > rows:
                df_temp = df_temp.iloc[:rows]
            df = df_temp.reset_index(drop=True)

        # === 3. Low Card ===
        elif m_type == 'low_card':
            ndv = params['ndv']
            i_start, i_end = params['int_range']
            pool_int = np.random.choice(np.arange(i_start, i_end), ndv, replace=False)
            df['col_int'] = np.random.choice(pool_int, rows)

            pool_str = params['varchar_range']['options'][:ndv]
            df['col_varchar'] = np.random.choice(pool_str, rows)

            d_start, d_end = params['date_range']
            pool_date = self._generate_date_pool(d_start, d_end, ndv)
            df['col_datetime'] = np.random.choice(pool_date, rows)

        # === 4. TPCC ===
        elif m_type == 'tpcc':
            table = params.get('table_name', 'Order')
            if table == 'Order':
                df['col_int'] = np.arange(1, rows + 1)
                df['col_varchar'] = [fake.text(max_nb_chars=20) for _ in range(rows)]
                df['col_datetime'] = [datetime.now() - timedelta(seconds=np.random.randint(0, 100000)) for _ in
                                      range(rows)]
            else:
                df['col_int'] = np.arange(1, rows + 1)
                df['col_datetime'] = [datetime.now() for _ in range(rows)]
                df['col_varchar'] = ["mock" for _ in range(rows)]

        # === 5. TPCH ===
        elif m_type == 'tpch':
            table = params.get('table_name', 'LineItem')
            if table == 'LineItem':
                df['col_int'] = np.random.randint(1, rows // 4, rows)
                df['col_varchar'] = [fake.sentence() for _ in range(rows)]
                start_date = datetime(1992, 1, 1)
                days_range = (datetime(1998, 12, 1) - start_date).days
                df['col_datetime'] = [start_date + timedelta(days=random.randint(0, days_range)) for _ in range(rows)]
            else:
                df['col_int'] = np.arange(1, rows + 1)
                df['col_datetime'] = [datetime.now() for _ in range(rows)]
                df['col_varchar'] = ["mock" for _ in range(rows)]

        df = df.sample(frac=1).reset_index(drop=True)
        return df


class DataModifier:
    def __init__(self, sql_generator=None):
        self.sql_gen = sql_generator

    def apply_changes(self, df, model_config, model_name):
        inc_params = model_config.get('incremental', {})
        print(f"    > Applying incremental logic to: {model_name}")

        target_cols = [c for c in df.columns if c.startswith(f"{model_name}_")]
        if not target_cols:
            return df

        id_col_name = f"{model_name}_int"

        # 1. DELETE
        if inc_params.get('delete_ratio', 0) > 0:
            valid_indices = df[df[target_cols[0]].notna()].index
            del_count = int(len(valid_indices) * inc_params['delete_ratio'])
            if del_count > 0:
                drop_indices = np.random.choice(valid_indices, del_count, replace=False)

                # Log SQL
                if self.sql_gen and id_col_name in df.columns:
                    ids_to_delete = df.loc[drop_indices, id_col_name].values
                    self.sql_gen.log_delete(model_name, id_col_name, ids_to_delete)

                # Apply
                df.loc[drop_indices, target_cols] = np.nan
                print(f"      - Delete: Set {del_count} rows to NaN.")

        # 2. UPDATE
        if inc_params.get('update_ratio', 0) > 0:
            valid_indices = df[df[target_cols[0]].notna()].index
            if len(valid_indices) > 0:
                upd_count = int(len(valid_indices) * inc_params['update_ratio'])

                if upd_count > 0:
                    upd_indices = np.random.choice(valid_indices, upd_count, replace=True)

                    # Prepare config for regeneration
                    temp_config = copy.deepcopy(model_config)
                    temp_config['params']['rows'] = upd_count

                    # Apply Overrides
                    overridden_keys = []
                    for k, v in inc_params.items():
                        if k not in CONTROL_KEYS:
                            temp_config['params'][k] = v
                            overridden_keys.append(k)

                    if overridden_keys:
                        print(
                            f"      - Update: Regenerating {upd_count} rows with OVERRIDDEN params: {overridden_keys}")
                    else:
                        print(f"      - Update: Regenerating {upd_count} rows with BASE params.")

                    gen = DataGenerator()
                    df_update = gen.generate(temp_config)

                    rename_map = {
                        'col_int': f'{model_name}_int',
                        'col_datetime': f'{model_name}_datetime',
                        'col_varchar': f'{model_name}_varchar'
                    }
                    df_update = df_update.rename(columns=rename_map)

                    actual_update_cols = [c for c in rename_map.values() if c in df.columns]

                    # Log SQL
                    if self.sql_gen and id_col_name in df.columns:
                        old_ids = df.loc[upd_indices, id_col_name].values
                        new_values = df_update[actual_update_cols].values
                        self.sql_gen.log_update(model_name, id_col_name, old_ids, actual_update_cols, new_values)

                    # Apply
                    df.loc[upd_indices, actual_update_cols] = df_update[actual_update_cols].values
                    print(f"      - Update: Modified {upd_count} rows.")

        # 3. INSERT
        insert_rows = inc_params.get('insert_rows', 0)
        if insert_rows > 0:
            temp_config = copy.deepcopy(model_config)
            temp_config['params']['rows'] = insert_rows

            overridden_keys = []
            for k, v in inc_params.items():
                if k not in CONTROL_KEYS:
                    temp_config['params'][k] = v
                    overridden_keys.append(k)

            if overridden_keys:
                print(f"      - Insert: Generating {insert_rows} rows with OVERRIDDEN params: {overridden_keys}")
            else:
                print(f"      - Insert: Generating {insert_rows} rows with BASE params.")

            gen = DataGenerator()
            df_insert = gen.generate(temp_config)

            rename_map = {
                'col_int': f'{model_name}_int',
                'col_datetime': f'{model_name}_datetime',
                'col_varchar': f'{model_name}_varchar'
            }
            df_insert = df_insert.rename(columns=rename_map)

            # Log SQL
            if self.sql_gen:
                self.sql_gen.log_insert(model_name, df_insert)

            # Apply
            df = pd.concat([df, df_insert], ignore_index=True)

        return df


class DataAnalyzer:
    def analyze(self, df, output_file):
        summary_rows = []
        for col in df.columns:
            parts = col.rsplit('_', 1)
            if len(parts) == 2:
                model_name, col_type = parts[0], parts[1]
            else:
                model_name, col_type = "Unknown", col

            col_data = df[col]
            valid_count = col_data.count()

            stats = {
                "Model": model_name,
                "Type": col_type,
                "Column": col,
                "Total Rows": len(df),
                "Valid Rows": valid_count,
                "NDV": col_data.nunique(),
                "Null Count": col_data.isnull().sum(),
            }

            if valid_count > 0:
                is_numeric = pd.api.types.is_numeric_dtype(col_data)
                is_date = pd.api.types.is_datetime64_any_dtype(col_data)
                if is_numeric or is_date:
                    stats["Range"] = f"[{col_data.min()}, {col_data.max()}]"
                else:
                    try:
                        valid_s = col_data.dropna().astype(str)
                        stats["Range"] = f"[{valid_s.min()}, {valid_s.max()}]"
                    except:
                        stats["Range"] = "N/A"
                vc = col_data.value_counts(normalize=True)
                stats["Top 10 (Val:Ratio)"] = vc.head(10).to_dict()
            else:
                stats["Range"] = "Empty"
                stats["Top 10 (Val:Ratio)"] = {}

            summary_rows.append(stats)
        res_df = pd.DataFrame(summary_rows)
        res_df.to_csv(output_file, index=False)
        return res_df


def rename_columns(df, model_name):
    mapping = {
        'col_int': f'{model_name}_int',
        'col_datetime': f'{model_name}_datetime',
        'col_varchar': f'{model_name}_varchar'
    }
    return df.rename(columns=mapping)


def parse_arguments():
    parser = argparse.ArgumentParser(description="DataSet Builder V10")
    parser.add_argument('--mode', type=str, choices=['base', 'incremental'], required=True)
    parser.add_argument('--sql-file', type=str, default='incremental_changes.sql',
                        help="Output file for SQL statements")
    return parser.parse_args()


def main():
    args = parse_arguments()
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: 'config.json' not found.")
        sys.exit(1)

    # 在增量模式下初始化 SQL 生成器
    sql_gen = None
    if args.mode == 'incremental':
        sql_gen = SqlGenerator()

    gen = DataGenerator()
    mod = DataModifier(sql_generator=sql_gen)
    analyzer = DataAnalyzer()

    # ----------------------------------------------------
    # MODE: BASE
    # ----------------------------------------------------
    if args.mode == 'base':
        print(f"\n>>> [BASE MODE] Generating initial data...")
        for model_cfg in config['models']:
            name = model_cfg['name']
            m_type = model_cfg['type']
            print(f"  Processing Model: [{name}] (Type: {m_type})")

            # 生成独立数据
            df_model = gen.generate(model_cfg)
            df_model = rename_columns(df_model, name)

            # 保存
            filename = f"dataset_{name}_base.csv"
            df_model.to_csv(filename, index=False)

            # 分析
            analyzer.analyze(df_model, f"summary_{name}_base.csv")

    # ----------------------------------------------------
    # MODE: INCREMENTAL
    # ----------------------------------------------------
    elif args.mode == 'incremental':
        print(f"\n>>> [INCREMENTAL MODE] Applying changes...")

        for model_cfg in config['models']:
            name = model_cfg['name']
            base_file = f"dataset_{name}_base.csv"

            if os.path.exists(base_file):
                print(f"  Loading table: {base_file}")
                # 读取 Base
                df_model = pd.read_csv(base_file)
                # 转换日期
                for col in df_model.columns:
                    if 'datetime' in col: df_model[col] = pd.to_datetime(df_model[col], errors='coerce')

                # 应用变更
                df_model = mod.apply_changes(df_model, model_cfg, name)

                # 保存 Incremental
                outfile = f"dataset_{name}_incremental.csv"
                df_model.to_csv(outfile, index=False)

                # 分析
                analyzer.analyze(df_model, f"summary_{name}_incremental.csv")
            else:
                print(f"  Warning: Base file {base_file} not found. Skipping.")

        # 保存 SQL
        if sql_gen:
            print(f">>> Saving SQL changes to {args.sql_file}...")
            sql_gen.save_to_file(args.sql_file)


if __name__ == "__main__":
    main()