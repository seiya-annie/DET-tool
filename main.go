package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	datapkg "det-tool/internal/data"
	dbpkg "det-tool/internal/db"
	querypkg "det-tool/internal/query"
	reportpkg "det-tool/internal/report"
	itypes "det-tool/internal/types"

	"github.com/spf13/pflag"
	"strings"
)

const (
	INTERNAL_MODELS     = "skew,holes,low_card,partition_skew"
	EXTERNAL_MODELS     = "external_tpcc,external_tpch"
	TARGET_QUERY_MODELS = "skew,holes,low_card,partition_skew"
	CONTROL_KEYS        = "insert_rows,update_ratio,delete_ratio"
)

// Alias core types to internal/types so existing references remain valid
type Config = itypes.Config
type ModelConfig = itypes.ModelConfig
type DBConfig = itypes.DBConfig
type QueryResult = itypes.QueryResult

var (
	all                   bool
	genBase               bool
	genInc                bool
	genQuery              bool
	execQuery             bool
	sqlFile               string
	configFile            string
	dbConfigFile          string
	analyzeWaitRetries    int
	analyzeWaitIntervalMs int
	dmlBatchSize          int
	insertBatchSize       int
	incInsertMode         string
	outputDir             string
	tmpDir                string
	reportUseActual       bool
)

func init() {
	pflag.BoolVar(&all, "all", false, "Execute all steps: Base -> Query -> Inc -> Exec Query")
	pflag.BoolVar(&genBase, "gen-base", false, "Step 1: Generate & Load Base Data")
	pflag.BoolVar(&genInc, "gen-inc", false, "Step 2: Generate & Execute Incremental Data")
	pflag.BoolVar(&genQuery, "gen-query", false, "Step 3: Generate SQL Queries (based on current DB stats)")
	pflag.BoolVar(&execQuery, "exec-query", false, "Step 4: Execute SQL Queries & Report")
	pflag.StringVar(&sqlFile, "sql-file", "incremental_dml.sql", "File for incremental DML")
	pflag.StringVar(&configFile, "config", "config.json", "Configuration file for models")
	pflag.StringVar(&dbConfigFile, "db-config", "db_config.json", "Database configuration file")
	pflag.IntVar(&analyzeWaitRetries, "analyze-retries", 20, "Max retries waiting for stats healthy after ANALYZE")
	pflag.IntVar(&analyzeWaitIntervalMs, "analyze-interval-ms", 1000, "Interval (ms) between retries waiting for stats healthy")
	pflag.IntVar(&dmlBatchSize, "dml-batch-size", 100, "Statements per transaction commit when executing SQL file")
	pflag.IntVar(&insertBatchSize, "insert-batch-size", 1000, "Rows per multi-row INSERT in incremental DML generation")
	pflag.StringVar(&incInsertMode, "inc-insert-mode", "insert", "Mode for incremental inserts: insert|load")
	pflag.StringVar(&outputDir, "output-dir", "output", "Base directory for generated outputs (reports, queries, datasets, DML file)")
	pflag.StringVar(&tmpDir, "tmp-dir", "tmp", "Directory for temporary files (incremental CSVs)")
	pflag.BoolVar(&reportUseActual, "report-use-actual-inc", false, "Display actual modify ratio measured from incremental DML instead of target from config")
}

func main() {
	pflag.Parse()

	if all {
		genBase = true
		genInc = true
		genQuery = true
		execQuery = true
	}

	config, err := loadConfig(configFile)
	if err != nil {
		log.Fatalf("Config Error: %v", err)
	}

	dbConfig, err := loadDBConfig(dbConfigFile)
	if err != nil {
		log.Fatalf("DB Config Error: %v", err)
	}

	dbManager := dbpkg.NewDBManager(*dbConfig)
	// Apply ANALYZE wait policy from CLI flags
	if analyzeWaitRetries <= 0 {
		analyzeWaitRetries = 20
	}
	if analyzeWaitIntervalMs <= 0 {
		analyzeWaitIntervalMs = 1000
	}
	dbManager.SetAnalyzeWaitPolicy(analyzeWaitRetries, time.Duration(analyzeWaitIntervalMs)*time.Millisecond)
	if dmlBatchSize <= 0 {
		dmlBatchSize = 100
	}
	dbManager.SetDMLBatchSize(dmlBatchSize)
	// Prepare output directories
	queriesDir := filepath.Join(outputDir, "queries")
	reportsDir := filepath.Join(outputDir, "reports")
	datasetsDir := filepath.Join(outputDir, "datasets")
	_ = os.MkdirAll(queriesDir, 0755)
	_ = os.MkdirAll(reportsDir, 0755)
	_ = os.MkdirAll(datasetsDir, 0755)
	_ = os.MkdirAll(tmpDir, 0755)
	// Relocate default incremental DML file into output dir when not overridden
	if sqlFile == "incremental_dml.sql" {
		sqlFile = filepath.Join(outputDir, sqlFile)
	}
	externalRunner := NewExternalBenchRunner(*dbConfig)
	// internal/data is used for generation; keep alias for clarity
	queryBuilder := querypkg.NewQueryBuilder()
	// [新增] 初始化 ReportGenerator
	reportGenerator := reportpkg.NewReportGenerator()

	// Accumulate actual modify ratios if gen-inc runs in this process
	actualRatios := make(map[string]float64)
	// Collect line-level change stats per model for Step2 logging
	type incStats struct{ Inserted, Updated, Deleted, BaseRows int }
	appliedStats := make(map[string]incStats)

	// Step 1: Base Data Generation
	if genBase {
		fmt.Println("\n=== [Step 1] Base Data Generation ===")
		dbManager.InitDB(true)
		dbManager.DisableAutoAnalyze()
		for _, model := range config.Models {
			name := model.Name
			if contains(EXTERNAL_MODELS, model.Type) {
				externalRunner.PrepareData(model)
			} else {
				fmt.Printf("Generating base data for %s...\n", name)
				// Build a minimal map for internal/data generator compatibility
				m := map[string]interface{}{
					"Name":        model.Name,
					"Type":        model.Type,
					"Params":      model.Params,
					"Incremental": model.Incremental,
				}
				df := datapkg.NewDataGenerator().Generate(m)
				csvPath := filepath.Join(datasetsDir, fmt.Sprintf("dataset_%s_base.csv", name))
				if err := saveDataFrameToCSV(df, csvPath); err != nil {
					log.Printf("Error saving CSV for %s: %v", name, err)
					continue
				}
				partClause := df.PartitionClause()
				dbManager.CreateTable(name, df, partClause)
				dbManager.LoadDataInfile(name, csvPath)
			}
		}

		// Analyze tables after creation
		time.Sleep(1 * time.Minute)
		for _, model := range config.Models {
			if !contains(EXTERNAL_MODELS, model.Type) {
				// Internal tables in default DB
				dbManager.AnalyzeTable(model.Name)
			} else {
				// External models (TPCC/TPCH): analyze all tables in their specific DB
				toolName := strings.Replace(model.Type, "external_", "", 1)
				targetDB := ""
				if v, ok := model.Params["db_name"]; ok {
					targetDB = fmt.Sprintf("%v", v)
				}
				if targetDB == "" {
					targetDB = toolName
				}
				dbManager.AnalyzeAllTablesInDB(targetDB)
			}
		}
		// Snapshot STATS_META after all ANALYZE
		dbManager.DumpStatsMetaForDB(dbConfig.DBName)
	}

	// Step 2: Incremental Data Generation & Execution
	if genInc {
		fmt.Println("\n=== [Step 2] Incremental Data Update ===")
		dbManager.InitDB(false)

		sqlGenerator := NewSqlGenerator()
		if insertBatchSize <= 0 {
			insertBatchSize = 1000
		}
		// Use internal/data modifier to build DML, then mirror to top-level DataFrame
		dataModifier2 := datapkg.NewDataModifier(insertBatchSize, tmpDir, dbManager)
		dataModifier2.SetInsertMode(incInsertMode)
		// SQL logger adapter to internal/data
		var logger datapkg.SQLLogger = &sqlLoggerAdapter{gen: sqlGenerator}

		// Pass 1: apply incremental DML for internal models first
		for _, model := range config.Models {
			if contains(EXTERNAL_MODELS, model.Type) {
				continue
			}
			name := model.Name
			baseCSV := filepath.Join(datasetsDir, fmt.Sprintf("dataset_%s_base.csv", name))
			if _, err := os.Stat(baseCSV); err == nil {
				fmt.Printf("Applying changes to %s...\n", name)
				df, err := loadDataFrameFromCSV(baseCSV)
				if err != nil {
					log.Printf("Error loading CSV for %s: %v", name, err)
					continue
				}
				if !dbManager.TableExists(name) {
					log.Printf("Table %s not found. Please run --gen-base first to create base data.", name)
					continue
				}
				// Map model to generic map for internal/data modifier
				m := map[string]interface{}{"Name": model.Name, "Type": model.Type, "Params": model.Params, "Incremental": model.Incremental}
				// Apply with internal/data modifier and logger
				df1, stats := dataModifier2.Apply(df, m, name, logger)
				// record actual modify ratio for report if enabled later
				if stats.BaseRows > 0 {
					ratio := float64(stats.Inserted)/float64(stats.BaseRows) + float64(stats.Updated)/float64(stats.BaseRows) + float64(stats.Deleted)/float64(stats.BaseRows)
					actualRatios[name] = ratio
				}
				// record raw line-level stats for Step2 summary logging
				appliedStats[name] = incStats{Inserted: stats.Inserted, Updated: stats.Updated, Deleted: stats.Deleted, BaseRows: stats.BaseRows}
				if err := saveDataFrameToCSV(df1, baseCSV); err != nil {
					log.Printf("Error saving modified CSV for %s: %v", name, err)
				}
			}
		}

		sqlGenerator.Save(sqlFile)
		if _, err := os.Stat(sqlFile); err == nil {
			if info, err := os.Stat(sqlFile); err == nil {
				fmt.Printf("Executing incremental DMLs from %s (size=%d bytes)...\n", sqlFile, info.Size())
				if info.Size() == 0 {
					fmt.Println("    [DB] File is empty, skip executing incremental DMLs.")
				} else {
					dbManager.ExecuteSQLFile(sqlFile)
				}
			} else {
				fmt.Printf("Executing incremental DMLs from %s...\n", sqlFile)
				dbManager.ExecuteSQLFile(sqlFile)
			}
		}

		// Pass 2: run external workloads (tpcc/tpch)
		for _, model := range config.Models {
			if contains(EXTERNAL_MODELS, model.Type) {
				externalRunner.RunWorkload(model)
			}
		}

		// Step 2 summary logging: line-level stats and SHOW STATS_META snapshot
		fmt.Println("\n--- [Step 2] Incremental Data Summary (line-level) ---")
		if len(appliedStats) == 0 {
			fmt.Println("No internal models were updated in this step.")
		} else {
			for name, s := range appliedStats {
				ratio := 0.0
				if s.BaseRows > 0 {
					ratio = float64(s.Inserted+s.Updated+s.Deleted) / float64(s.BaseRows)
				}
				fmt.Printf("Model=%s | BaseRows=%d, Inserted=%d, Updated=%d, Deleted=%d | ModifyRatio(line-level)=%.4f\n",
					name, s.BaseRows, s.Inserted, s.Updated, s.Deleted, ratio)
			}
		}

		// Allow TiDB stats to refresh before generating/executing queries
		if genQuery || execQuery {
			fmt.Println("Waiting 1 minutes for TiDB stats to refresh...")
			time.Sleep(2 * time.Minute)
		}
		// Print STATS_META for default DB to correlate key-level modify_count with line-level changes
		dbManager.DumpStatsMetaForDB(dbConfig.DBName)
	}

	// Step 3: Generate Queries (Based on CURRENT DB State)
	if genQuery {
		fmt.Println("\n=== [Step 3] Generate Queries (Adaptive) ===")
		dbManager.InitDB(false)
		types := make([]string, 0, len(config.Models))
		for _, model := range config.Models {
			types = append(types, model.Type)
		}
		fmt.Printf("Loaded model types: %s\n", strings.Join(types, ", "))
		for _, model := range config.Models {
			if contains(TARGET_QUERY_MODELS, model.Type) {
				name := model.Name
				cols := []string{fmt.Sprintf("%s_int", name), fmt.Sprintf("%s_datetime", name)}
				if strings.EqualFold(model.Type, "partition_skew") {
					cols = []string{"partition_skew_id", "partition_skew_datetime"}
				}
				stats := dbManager.GetTableStats(name, cols)

				outfile := filepath.Join(queriesDir, fmt.Sprintf("queries_%s.sql", name))
				queryBuilder.Generate(model, name, outfile, stats)
				fmt.Printf("Generated %s based on DB stats: %v\n", outfile, stats)
			}
		}
	}

	// Step 4: Execute Queries & Report (internal + external)
	if execQuery {
		fmt.Println("\n=== [Step 4] Execute Queries & Report ===")
		dbManager.InitDB(false)

		// Stats healthy aggregation:
		// - For internal models: use default DB's table health (keyed by table)
		// - For external models (tpcc/tpch): aggregate target DB tables and map to model name using worst (min) health
		statsHealthyInfo := map[string]int{}
		// internal DB tables
		internalHealth := dbManager.GetStatsHealthyForDB(dbConfig.DBName)
		for tbl, h := range internalHealth {
			statsHealthyInfo[tbl] = h
		}
		// external DBs
		for _, model := range config.Models {
			if contains(EXTERNAL_MODELS, model.Type) {
				toolName := strings.Replace(model.Type, "external_", "", 1)
				targetDB := ""
				if v, ok := model.Params["db_name"]; ok {
					targetDB = fmt.Sprintf("%v", v)
				}
				if targetDB == "" {
					targetDB = toolName
				}
				hmap := dbManager.GetStatsHealthyForDB(targetDB)
				worst := 100
				for _, v := range hmap {
					if v < worst {
						worst = v
					}
				}
				statsHealthyInfo[model.Name] = worst
			}
		}
		fmt.Printf("Stats healthy info: %v\n", statsHealthyInfo)

		var allResults []QueryResult
		for _, model := range config.Models {
			switch {
			case contains(TARGET_QUERY_MODELS, model.Type):
				// Internal generated workload
				name := model.Name
				qfile := filepath.Join(queriesDir, fmt.Sprintf("queries_%s.sql", name))
				fmt.Printf("Executing %s...\n", qfile)
				results := dbManager.ExecuteAndExplain(qfile)
				for i := range results {
					results[i].Model = name
				}
				allResults = append(allResults, results...)

			case contains(EXTERNAL_MODELS, model.Type):
				// External benchmarks: run a curated set of read-only queries on target DB
				toolName := strings.Replace(model.Type, "external_", "", 1)
				targetDB := ""
				if v, ok := model.Params["db_name"]; ok {
					targetDB = fmt.Sprintf("%v", v)
				}
				var queries []string
				label := model.Name

				if toolName == "tpch" {
					var ids []string
					if inc := model.Incremental; inc != nil {
						if arr, ok := inc["queries"].([]interface{}); ok {
							for _, q := range arr {
								if s, ok := q.(string); ok {
									ids = append(ids, s)
								}
							}
						}
					}
					queries = GetTPCHQueries(ids)
				} else if toolName == "tpcc" {
					queries = GetTPCCQueries()
				}

				if targetDB == "" {
					// Fall back to tool name (align with external runner default)
					targetDB = toolName
				}

				if len(queries) > 0 && targetDB != "" {
					fmt.Printf("Executing %s curated queries on DB '%s'...\n", toolName, targetDB)
					results := dbManager.ExecuteAndExplainQueriesOnDB(targetDB, queries)
					for i := range results {
						results[i].Model = label
					}
					allResults = append(allResults, results...)
				} else {
					fmt.Printf("No queries prepared for external model %s or missing db_name.\n", model.Name)
				}
			}
		}

		if len(allResults) > 0 {
			ts := time.Now().Format("20060102_150405")
			csvName := filepath.Join(reportsDir, fmt.Sprintf("report_execution_%s.csv", ts))
			htmlName := filepath.Join(reportsDir, fmt.Sprintf("report_execution_%s.html", ts))
			jsonName := filepath.Join(reportsDir, fmt.Sprintf("report_execution_%s.json", ts))

			// [修改] 使用 reportGenerator 并传入 statsHealthyInfo + actual/flag
			if err := reportpkg.NewReportGenerator().GenerateCSVReport(allResults, csvName, config, statsHealthyInfo, actualRatios, reportUseActual); err != nil {
				log.Printf("Error generating CSV report: %v", err)
			}
			if err := reportGenerator.GenerateHTMLReport(allResults, htmlName, config, statsHealthyInfo, actualRatios, reportUseActual); err != nil {
				log.Printf("Error generating HTML report: %v", err)
			}
			if err := reportGenerator.GenerateJSONReport(allResults, jsonName, config, statsHealthyInfo, actualRatios, reportUseActual); err != nil {
				log.Printf("Error generating JSON report: %v", err)
			}

			// [修改] 使用 reportGenerator 的方法显示 Top Queries
			reportGenerator.DisplayTopQueries(allResults, 10)
		} else {
			fmt.Println("No queries executed or no results found.")
		}
	}
}

func loadConfig(filename string) (*Config, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}
	return &config, nil
}

func loadDBConfig(filename string) (*DBConfig, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var config DBConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}
	return &config, nil
}

func contains(list string, item string) bool {
	if list == "" {
		return false
	}
	for _, v := range strings.Split(list, ",") {
		if v == item {
			return true
		}
	}
	return false
}

func saveDataFrameToCSV(df *datapkg.DataFrame, path string) error { return df.SaveCSV(path) }
func loadDataFrameFromCSV(path string) (*datapkg.DataFrame, error) {
	return datapkg.LoadDataFrameFromCSV(path)
}

// derivePartitionClause builds partition clause for known partitioned models.
func derivePartitionClause(m ModelConfig) string {
	if strings.ToLower(m.Type) != "partition_skew" {
		return ""
	}
	// Reuse generator logic: build month starts from date_range (fallback to 6 months).
	start, end := "", ""
	if dr, ok := m.Params["date_range"].([]interface{}); ok {
		if len(dr) > 0 {
			start = fmt.Sprintf("%v", dr[0])
		}
		if len(dr) > 1 {
			end = fmt.Sprintf("%v", dr[1])
		}
	}
	startTime, _ := time.Parse("2006-01-02", start)
	if startTime.IsZero() {
		now := time.Now()
		startTime = time.Date(now.Year(), 1, 1, 0, 0, 0, 0, time.UTC)
	}
	endTime, _ := time.Parse("2006-01-02", end)
	if endTime.IsZero() {
		endTime = startTime.AddDate(0, 6, 0)
	}
	startTime = time.Date(startTime.Year(), startTime.Month(), 1, 0, 0, 0, 0, time.UTC)
	endTime = time.Date(endTime.Year(), endTime.Month(), 1, 0, 0, 0, 0, time.UTC)
	if !endTime.After(startTime) {
		endTime = startTime.AddDate(0, 1, 0)
	}
	monthStarts := []time.Time{}
	for cur := startTime; cur.Before(endTime) || cur.Equal(endTime); cur = cur.AddDate(0, 1, 0) {
		monthStarts = append(monthStarts, cur)
		if cur.Year() == endTime.Year() && cur.Month() == endTime.Month() {
			break
		}
	}
	return datapkg.BuildMonthlyPartitionClause(monthStarts, "partition_skew_datetime")
}

// [已删除] 旧的生成报告辅助函数 (generateCSVReport, generateHTMLReport 等)，因为现在使用 reporter.go

// computeModelTargetRatio computes insert/base + update + delete using config values.
func computeModelTargetRatio(m ModelConfig) float64 {
	params := m.Params
	inc := m.Incremental
	if inc == nil {
		return 0
	}
	baseRows := 0.0
	if v, ok := params["rows"]; ok {
		switch x := v.(type) {
		case float64:
			baseRows = x
		case int:
			baseRows = float64(x)
		case string:
			var f float64
			fmt.Sscanf(x, "%f", &f)
			baseRows = f
		}
	}
	if baseRows <= 0 {
		baseRows = 1000
	}
	getF := func(key string) float64 {
		if val, ok := inc[key]; ok {
			switch t := val.(type) {
			case float64:
				return t
			case int:
				return float64(t)
			case string:
				var f float64
				fmt.Sscanf(t, "%f", &f)
				return f
			}
		}
		return 0
	}
	insertRows := getF("insert_rows")
	updateRatio := getF("update_ratio")
	deleteRatio := getF("delete_ratio")
	return (insertRows / baseRows) + updateRatio + deleteRatio
}

// computeAvgTargetRatio computes the average target modify ratio for non-external models.
func computeAvgTargetRatio(cfg *Config) float64 {
	var sum float64
	var cnt int
	for _, m := range cfg.Models {
		if strings.HasPrefix(m.Type, "external_") {
			continue
		}
		r := computeModelTargetRatio(m)
		if r > 0 {
			sum += r
			cnt++
		}
	}
	if cnt == 0 {
		return 0
	}
	return sum / float64(cnt)
}
