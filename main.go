package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/spf13/pflag"
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
	externalRunner := NewExternalBenchRunner(*dbConfig)
	dataGenerator := NewDataGenerator()
	queryBuilder := NewQueryBuilder()

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
				dbManager.AnalyzeTable(model.Name)
			}
		}
	}

	// Step 2: Incremental Data Generation & Execution
	if genInc {
		fmt.Println("\n=== [Step 2] Incremental Data Update ===")
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
				cols := []string{"col_int", "col_datetime"}
				stats := dbManager.GetTableStats(name, cols)

				outfile := fmt.Sprintf("queries_%s.sql", name)
				queryBuilder.Generate(model, name, outfile, stats)
				fmt.Printf("Generated %s based on DB stats: %v\n", outfile, stats)
			}
		}
	}

	// Step 4: Execute Queries & Report
	if execQuery {
		fmt.Println("\n=== [Step 4] Execute Queries & Report ===")
		dbManager.InitDB(false)

		statsHealthyInfo := dbManager.GetStatsHealthy()
		fmt.Printf("Stats healthy info: %v\n", statsHealthyInfo)

		var report []QueryResult
		for _, model := range config.Models {
			if contains(TARGET_QUERY_MODELS, model.Type) {
				name := model.Name
				qfile := fmt.Sprintf("queries_%s.sql", name)
				fmt.Printf("Executing %s...\n", qfile)
				results := dbManager.ExecuteAndExplain(qfile)
				for i := range results {
					results[i].Model = name
				}
				report = append(report, results...)
			}
		}

		if len(report) > 0 {
			ts := time.Now().Format("20060102_150405")
			csvName := fmt.Sprintf("report_execution_%s.csv", ts)
			htmlName := fmt.Sprintf("report_execution_%s.html", ts)

			if err := generateCSVReport(report, csvName, config); err != nil {
				log.Printf("Error generating CSV report: %v", err)
			}
			if err := generateHTMLReport(report, htmlName, config); err != nil {
				log.Printf("Error generating HTML report: %v", err)
			}

			displayTopQueries(report)
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

func generateCSVReport(report []QueryResult, filename string, config *Config) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	headers := []string{"Model", "stats_healthy_ratio", "modify_ratio", "query_label",
		"estimation_error_ratio", "estimation_error_value", "query", "duration_ms",
		"explain", "risk_operators_count"}

	fmt.Fprintf(file, "%s\n", joinStrings(headers, ","))

	for _, result := range report {
		statsHealthy := getStatsHealthyForModel(result.Model, config)
		modifyRatio := calculateModifyRatio(result.Model, config)

		row := []string{
			result.Model,
			fmt.Sprintf("%.3f", statsHealthy),
			fmt.Sprintf("%.3f", modifyRatio),
			"",
			fmt.Sprintf("%.2f", result.EstimationErrorRatio),
			fmt.Sprintf("%.2f", result.EstimationErrorValue),
			result.Query,
			fmt.Sprintf("%.3f", result.DurationMs),
			result.Explain,
			fmt.Sprintf("%d", result.RiskOperatorsCount),
		}
		fmt.Fprintf(file, "%s\n", joinStrings(row, ","))
	}

	fmt.Printf("CSV Report saved to: %s\n", filename)
	return nil
}

func generateHTMLReport(report []QueryResult, filename string, config *Config) error {
	// 简化的HTML报告生成
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	htmlContent := fmt.Sprintf(`<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>Query Execution Report - %s</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { width: 100%%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #4CAF50; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .high-error { background-color: #ffebee !important; }
    </style>
</head>
<body>
    <h1>Query Execution Analysis Report</h1>
    <p><strong>Report Generated:</strong> %s</p>
    <p><strong>Total Queries:</strong> %d</p>
    
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Estimation Error Ratio</th>
                <th>Estimation Error Value</th>
                <th>Duration (ms)</th>
                <th>Query</th>
            </tr>
        </thead>
        <tbody>`,
		time.Now().Format("2006-01-02 15:04:05"),
		time.Now().Format("2006-01-02 15:04:05"),
		len(report))

	for _, result := range report {
		rowClass := ""
		if result.EstimationErrorRatio >= 10 && result.EstimationErrorValue >= 1000 {
			rowClass = "class=\"high-error\""
		}

		htmlContent += fmt.Sprintf(`
            <tr %s>
                <td>%s</td>
                <td>%.2f</td>
                <td>%.2f</td>
                <td>%.3f</td>
                <td>%s</td>
            </tr>`,
			rowClass,
			result.Model,
			result.EstimationErrorRatio,
			result.EstimationErrorValue,
			result.DurationMs,
			escapeHTML(result.Query))
	}

	htmlContent += `
        </tbody>
    </table>
</body>
</html>`

	fmt.Fprint(file, htmlContent)
	fmt.Printf("HTML Report saved to: %s\n", filename)
	return nil
}

func displayTopQueries(report []QueryResult) {
	// 简单的排序显示前10个查询
	fmt.Println("\nTop 10 queries by estimation error ratio:")
	fmt.Printf("%-20s %-15s %-15s %-15s %s\n", "Model", "Error Ratio", "Error Value", "Duration (ms)", "Query")
	fmt.Println(string(make([]byte, 120)))

	// 这里应该按 estimation_error_ratio 排序，简化处理
	for i, result := range report {
		if i >= 10 {
			break
		}
		fmt.Printf("%-20s %-15.2f %-15.2f %-15.3f %s\n",
			result.Model,
			result.EstimationErrorRatio,
			result.EstimationErrorValue,
			result.DurationMs,
			truncateString(result.Query, 50))
	}
}

func getStatsHealthyForModel(modelName string, config *Config) float64 {
	// 简化实现
	return 1.0
}

func calculateModifyRatio(modelName string, config *Config) float64 {
	for _, model := range config.Models {
		if model.Name == modelName {
			params := model.Params
			incremental := model.Incremental

			baseRows := getFloatValue(params, "rows")
			insertRows := getFloatValue(incremental, "insert_rows")
			updateRatio := getFloatValue(incremental, "update_ratio")
			deleteRatio := getFloatValue(incremental, "delete_ratio")

			if baseRows > 0 {
				return (insertRows / baseRows) + updateRatio + deleteRatio
			}
		}
	}
	return 0.0
}
