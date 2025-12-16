package main

import (
    "encoding/json"
    "fmt"
    "log"
    "os"
    "time"

    "github.com/spf13/pflag"
    "strings"
)

const (
	INTERNAL_MODELS     = "skew,holes,low_card"
	EXTERNAL_MODELS     = "external_tpcc,external_tpch"
	TARGET_QUERY_MODELS = "skew,holes,low_card"
	CONTROL_KEYS        = "insert_rows,update_ratio,delete_ratio"
)

type Config struct {
	Models []ModelConfig `json:"models"`
}

type ModelConfig struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Type        string                 `json:"type"`
	Params      map[string]interface{} `json:"params"`
	Incremental map[string]interface{} `json:"incremental"`
}

type DBConfig struct {
	Host     string `json:"host"`
	Port     int    `json:"port"`
	User     string `json:"user"`
	Password string `json:"password"`
	DBName   string `json:"db_name"`
	Charset  string `json:"charset"`
}

type QueryResult struct {
    QueryID              int     `json:"query_id"`
    Query                string  `json:"query"`
    QueryLabel           string  `json:"query_label"`
    DurationMs           float64 `json:"duration_ms"`
    Explain              string  `json:"explain"`
    EstimationErrorValue float64 `json:"estimation_error_value"`
    EstimationErrorRatio float64 `json:"estimation_error_ratio"`
    RiskOperatorsCount   int     `json:"risk_operators_count"`
	Model                string  `json:"model"`
}

var (
    all          bool
    genBase      bool
    genInc       bool
    genQuery     bool
    execQuery    bool
    sqlFile      string
    configFile   string
    dbConfigFile string
    analyzeWaitRetries int
    analyzeWaitIntervalMs int
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

    dbManager := NewDBManager(*dbConfig)
    // Apply ANALYZE wait policy from CLI flags
    if analyzeWaitRetries <= 0 { analyzeWaitRetries = 20 }
    if analyzeWaitIntervalMs <= 0 { analyzeWaitIntervalMs = 1000 }
    dbManager.SetAnalyzeWaitPolicy(analyzeWaitRetries, time.Duration(analyzeWaitIntervalMs)*time.Millisecond)
	externalRunner := NewExternalBenchRunner(*dbConfig)
	dataGenerator := NewDataGenerator()
	queryBuilder := NewQueryBuilder()
	// [新增] 初始化 ReportGenerator
	reportGenerator := NewReportGenerator()

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
				df := dataGenerator.Generate(model)
				fmt.Println(">>>>>> ", df.columns)
				csvPath := fmt.Sprintf("dataset_%s_base.csv", name)
				if err := saveDataFrameToCSV(df, csvPath); err != nil {
					log.Printf("Error saving CSV for %s: %v", name, err)
					continue
				}
				dbManager.CreateTable(name, df)
				dbManager.LoadDataInfile(name, csvPath)
			}
		}

        // Analyze tables after creation
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
    }

	// Step 2: Incremental Data Generation & Execution
	if genInc {
		fmt.Println("\n=== [Step 2] Incremental Data Update ===")
		dbManager.InitDB(false)

		sqlGenerator := NewSqlGenerator()
		dataModifier := NewDataModifier(sqlGenerator)

		for _, model := range config.Models {
			name := model.Name
			if contains(EXTERNAL_MODELS, model.Type) {
				externalRunner.RunWorkload(model)
			} else {
				baseCSV := fmt.Sprintf("dataset_%s_base.csv", name)
				if _, err := os.Stat(baseCSV); err == nil {
					fmt.Printf("Applying changes to %s...\n", name)
					df, err := loadDataFrameFromCSV(baseCSV)
					if err != nil {
						log.Printf("Error loading CSV for %s: %v", name, err)
						continue
					}
					modifiedDF := dataModifier.Apply(df, model, name)
					if err := saveDataFrameToCSV(modifiedDF, baseCSV); err != nil {
						log.Printf("Error saving modified CSV for %s: %v", name, err)
					}
				}
			}
		}

		sqlGenerator.Save(sqlFile)
		if _, err := os.Stat(sqlFile); err == nil {
			fmt.Printf("Executing incremental DMLs from %s...\n", sqlFile)
			dbManager.ExecuteSQLFile(sqlFile)
		}
	}

	// Step 3: Generate Queries (Based on CURRENT DB State)
	if genQuery {
		fmt.Println("\n=== [Step 3] Generate Queries (Adaptive) ===")
		dbManager.InitDB(false)
		for _, model := range config.Models {
			if contains(TARGET_QUERY_MODELS, model.Type) {
				name := model.Name
				cols := []string{fmt.Sprintf("%s_int", name), fmt.Sprintf("%s_datetime", name)}
				stats := dbManager.GetTableStats(name, cols)

				outfile := fmt.Sprintf("queries_%s.sql", name)
				queryBuilder.Generate(model, name, outfile, stats)
				fmt.Printf("Generated %s based on DB stats: %v\n", outfile, stats)
			}
		}
	}

    // Step 4: Execute Queries & Report (internal + external)
    if execQuery {
        fmt.Println("\n=== [Step 4] Execute Queries & Report ===")
        dbManager.InitDB(false)

		// [修改] 获取 Stats Healthy (现在返回 map[string]int)
		statsHealthyInfo := dbManager.GetStatsHealthy()
		fmt.Printf("Stats healthy info: %v\n", statsHealthyInfo)

        var report []QueryResult
        for _, model := range config.Models {
            switch {
            case contains(TARGET_QUERY_MODELS, model.Type):
                // Internal generated workload
                name := model.Name
                qfile := fmt.Sprintf("queries_%s.sql", name)
                fmt.Printf("Executing %s...\n", qfile)
                results := dbManager.ExecuteAndExplain(qfile)
                for i := range results {
                    results[i].Model = name
                }
                report = append(report, results...)

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
                    report = append(report, results...)
                } else {
                    fmt.Printf("No queries prepared for external model %s or missing db_name.\n", model.Name)
                }
            }
        }

		if len(report) > 0 {
			ts := time.Now().Format("20060102_150405")
			csvName := fmt.Sprintf("report_execution_%s.csv", ts)
			htmlName := fmt.Sprintf("report_execution_%s.html", ts)
			jsonName := fmt.Sprintf("report_execution_%s.json", ts)

			// [修改] 使用 reportGenerator 并传入 statsHealthyInfo
			if err := reportGenerator.GenerateCSVReport(report, csvName, config, statsHealthyInfo); err != nil {
				log.Printf("Error generating CSV report: %v", err)
			}
			if err := reportGenerator.GenerateHTMLReport(report, htmlName, config, statsHealthyInfo); err != nil {
				log.Printf("Error generating HTML report: %v", err)
			}
			if err := reportGenerator.GenerateJSONReport(report, jsonName, config, statsHealthyInfo); err != nil {
				log.Printf("Error generating JSON report: %v", err)
			}

			// [修改] 使用 reportGenerator 的方法显示 Top Queries
			reportGenerator.DisplayTopQueries(report, 10)
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
	items := splitString(list, ",")
	for _, v := range items {
		if v == item {
			return true
		}
	}
	return false
}

func saveDataFrameToCSV(df *DataFrame, path string) error {
	return df.SaveCSV(path)
}

func loadDataFrameFromCSV(path string) (*DataFrame, error) {
	return LoadDataFrameFromCSV(path)
}

// [已删除] 旧的生成报告辅助函数 (generateCSVReport, generateHTMLReport 等)，因为现在使用 reporter.go
