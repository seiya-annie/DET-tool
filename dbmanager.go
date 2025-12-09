package main

import (
	"database/sql"
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	mysql "github.com/go-sql-driver/mysql"
)

// DBManager manages database operations
type DBManager struct {
	config DBConfig
	db     *sql.DB
}

// NewDBManager creates a new DBManager instance
func NewDBManager(config DBConfig) *DBManager {
	dbManager := &DBManager{config: config}
	dbManager.connect()
	return dbManager
}

// connect establishes database connection
func (dbm *DBManager) connect() {
	dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/?charset=%s&parseTime=true&loc=Local",
		dbm.config.User,
		dbm.config.Password,
		dbm.config.Host,
		dbm.config.Port,
		dbm.config.Charset)

	db, err := sql.Open("mysql", dsn)
	if err != nil {
		log.Fatalf("Error connecting to DB: %v", err)
	}

	// Test connection
	if err := db.Ping(); err != nil {
		log.Fatalf("Error pinging DB: %v", err)
	}

	dbm.db = db
	fmt.Printf(">>> Connected to Database: %s:%d\n", dbm.config.Host, dbm.config.Port)
}

// EnsureConnection ensures the database connection is active
func (dbm *DBManager) EnsureConnection() {
	if err := dbm.db.Ping(); err != nil {
		fmt.Println("    [DB] Reconnecting...")
		dbm.connect()
	}
}

// InitDB initializes the database
func (dbm *DBManager) InitDB(dropIfExists bool) {
	dbName := dbm.config.DBName

	if dropIfExists {
		// Drop database if exists
		_, err := dbm.db.Exec(fmt.Sprintf("DROP DATABASE IF EXISTS %s", dbName))
		if err != nil {
			log.Printf("Error dropping database: %v", err)
		}

		// Create database
		_, err = dbm.db.Exec(fmt.Sprintf("CREATE DATABASE IF NOT EXISTS %s", dbName))
		if err != nil {
			log.Printf("Error creating database: %v", err)
		}
	}

	// Use database
	_, err := dbm.db.Exec(fmt.Sprintf("USE %s", dbName))
	if err != nil {
		log.Printf("Error selecting database: %v", err)
	}

	fmt.Printf(">>> Selected Database: %s\n", dbName)
}

// DisableAutoAnalyze disables TiDB auto analyze
func (dbm *DBManager) DisableAutoAnalyze() {
	fmt.Println("    [DB] Disabling Global Auto Analyze...")
	_, err := dbm.db.Exec("SET GLOBAL tidb_enable_auto_analyze = OFF")
	if err != nil {
		fmt.Printf("    [Warning] Failed to disable auto analyze: %v\n", err)
	}
}

// CreateTable creates a table based on DataFrame structure
func (dbm *DBManager) CreateTable(tableName string, df *DataFrame) {
	if len(df.columns) == 0 {
		log.Printf("No columns found in DataFrame for table %s", tableName)
		return
	}

	// Build column definitions
	cols := []string{}
	indexes := []string{}

	for _, colName := range df.columns {
		sqlType := dbm.inferSQLType(df, colName)
		cols = append(cols, fmt.Sprintf("`%s` %s", colName, sqlType))
		indexes = append(indexes, fmt.Sprintf("KEY `idx_%s` (`%s`)", colName, colName))
	}

	// Add primary key
	cols = append([]string{"`id` bigint NOT NULL AUTO_INCREMENT"}, cols...)

	ddl := fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
		%s,
		%s,
		PRIMARY KEY (`+"`id`"+`)
	) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin`,
		tableName,
		strings.Join(cols, ", "),
		strings.Join(indexes, ", "))
	fmt.Println(ddl)

	// Drop table if exists
	_, err := dbm.db.Exec(fmt.Sprintf("DROP TABLE IF EXISTS %s", tableName))
	if err != nil {
		log.Printf("Error dropping table: %v", err)
	}

	// Create table
	_, err = dbm.db.Exec(ddl)
	if err != nil {
		log.Printf("Error creating table: %v", err)
	}

	fmt.Printf("    [DB] Table created: %s\n", tableName)
}

// inferSQLType infers SQL type from DataFrame column data
func (dbm *DBManager) inferSQLType(df *DataFrame, colName string) string {
	col := df.GetColumn(colName)
	if len(col) == 0 {
		return "VARCHAR(255)"
	}

	// Check first non-null value
	for _, val := range col {
		if val != nil {
			switch val.(type) {
			case int, int64, int32:
				return "BIGINT"
			case float64, float32:
				return "DOUBLE"
			case time.Time:
				return "DATETIME"
			default:
				return "VARCHAR(255)"
			}
		}
	}
	return "VARCHAR(255)"
}

// LoadDataInfile loads data from CSV file into table
func (dbm *DBManager) LoadDataInfile(tableName string, csvPath string) {
	absPath, err := filepath.Abs(csvPath)
	if err != nil {
		log.Printf("Error getting absolute path: %v", err)
		return
	}

	// 1. Read the CSV header to determine correct column mapping
	f, err := os.Open(absPath)
	if err != nil {
		log.Printf("    [Error] Failed to open CSV to read header: %v", err)
		return
	}

	// Create a temporary reader just to get the header
	csvReader := csv.NewReader(f)
	header, err := csvReader.Read()
	f.Close() // Close file immediately after reading header

	if err != nil {
		log.Printf("    [Error] Failed to read CSV header: %v", err)
		return
	}

	// Build the column list string, e.g., (`col_int`, `col_varchar`, `col_datetime`)
	// This ensures we map the CSV columns to the correct table columns and skip 'id'
	quotedCols := make([]string, len(header))
	for i, col := range header {
		quotedCols[i] = fmt.Sprintf("`%s`", strings.TrimSpace(col))
	}
	columnListSql := fmt.Sprintf("(%s)", strings.Join(quotedCols, ", "))

	// Replace backslashes for Windows
	absPath = strings.ReplaceAll(absPath, "\\", "/")
	mysql.RegisterLocalFile(absPath)

	// 2. Construct SQL
	// - Changed ENCLOSED BY '"' to OPTIONALLY ENCLOSED BY '"' to support unquoted CSVs
	// - Added columnListSql to explicitly map columns
	sql := fmt.Sprintf(`LOAD DATA LOCAL INFILE '%s' INTO TABLE %s 
		FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' 
		LINES TERMINATED BY '\n' 
		IGNORE 1 LINES 
		%s`, absPath, tableName, columnListSql)

	// Enable local infile for this session
	_, err = dbm.db.Exec("SET GLOBAL local_infile = 1")
	if err != nil {
		log.Printf("    [Warning] Failed to enable local_infile: %v", err)
	}

	_, err = dbm.db.Exec(sql)
	if err != nil {
		log.Printf("    [Error] Load Data failed: %v", err)
		return
	}

	fmt.Printf("    [DB] Data loaded into %s (Columns: %s)\n", tableName, strings.Join(header, ", "))
}

// GetSingleTableHealth gets stats health for a specific table
func (dbm *DBManager) GetSingleTableHealth(tableName string) int {
	dbName := dbm.config.DBName

	query := fmt.Sprintf("SHOW STATS_HEALTHY WHERE Db_name = '%s' AND Table_name = '%s'", dbName, tableName)
	rows, err := dbm.db.Query(query)
	if err != nil {
		return 0
	}
	defer rows.Close()

	var dbNameResult, tableNameResult string
	var healthy int

	if rows.Next() {
		err := rows.Scan(&dbNameResult, &tableNameResult, &healthy, &healthy)
		if err == nil {
			return healthy
		}
	}

	return 0
}

// AnalyzeTable analyzes a table and waits for health to reach 100
func (dbm *DBManager) AnalyzeTable(tableName string) {
	fmt.Printf("    [DB] Executing Manual Analyze: ANALYZE TABLE %s ALL COLUMNS ...\n", tableName)
	startTime := time.Now()

	_, err := dbm.db.Exec(fmt.Sprintf("ANALYZE TABLE %s ALL COLUMNS", tableName))
	if err != nil {
		fmt.Printf("    [Error] Analyze failed: %v\n", err)
		return
	}

	// Wait for stats to become healthy
	fmt.Println("    [DB] Waiting for stats to become healthy (100%)...")
	maxRetries := 20

	for i := 0; i < maxRetries; i++ {
		health := dbm.GetSingleTableHealth(tableName)
		if health == 100 {
			break
		}
		time.Sleep(1 * time.Second)

		if i == maxRetries-1 {
			fmt.Printf("    [Warning] Stats health reached %d%%, timed out waiting for 100%%.\n", health)
		}
	}

	duration := time.Since(startTime)
	finalHealth := dbm.GetSingleTableHealth(tableName)
	fmt.Printf("    [DB] Analyze finished in %.2fs (Health: %d%%)\n", duration.Seconds(), finalHealth)
}

// ExecuteSQLFile executes SQL statements from a file
func (dbm *DBManager) ExecuteSQLFile(sqlPath string) {
	fmt.Printf("    [DB] Executing SQL script: %s\n", sqlPath)

	if _, err := os.Stat(sqlPath); os.IsNotExist(err) {
		return
	}

	content, err := os.ReadFile(sqlPath)
	if err != nil {
		log.Printf("Error reading SQL file: %v", err)
		return
	}

	statements := strings.Split(string(content), ";")

	for _, statement := range statements {
		statement = strings.TrimSpace(statement)
		if statement == "" {
			continue
		}

		_, err := dbm.db.Exec(statement)
		if err != nil {
			fmt.Printf("      SQL Error: %v\n", err)
		}
	}
}

// ExecuteAndExplain executes queries and returns results with explain plans
func (dbm *DBManager) ExecuteAndExplain(queryFile string) []QueryResult {
	if _, err := os.Stat(queryFile); os.IsNotExist(err) {
		return []QueryResult{}
	}

	content, err := os.ReadFile(queryFile)
	if err != nil {
		log.Printf("Error reading query file: %v", err)
		return []QueryResult{}
	}

	queries := strings.Split(string(content), ";")
	results := []QueryResult{}
	queryID := 1

	for _, query := range queries {
		query = strings.TrimSpace(query)
		if query == "" || strings.HasPrefix(query, "--") {
			continue
		}

		// Execute query and measure time
		start := time.Now()
		rows, err := dbm.db.Query(query)
		if err != nil {
			fmt.Printf("      Q%d Error: %v\n", queryID, err)
			queryID++
			continue
		}
		rows.Close()
		duration := time.Since(start).Milliseconds()

		// Get explain analyze
		explainQuery := fmt.Sprintf("EXPLAIN analyze %s", query)
		explainRows, err := dbm.db.Query(explainQuery)
		if err != nil {
			fmt.Printf("      Explain Error: %v\n", err)
			queryID++
			continue
		}
		defer explainRows.Close()

		// Build explain result
		explainResult := ""
		for explainRows.Next() {
			var row string
			if err := explainRows.Scan(&row); err == nil {
				explainResult += row + "\n"
			}
		}

		// Parse explain analyze
		estErrorValue, estErrorRatio, riskCount := dbm.parseExplainAnalyze(explainResult)

		result := QueryResult{
			QueryID:              queryID,
			Query:                query,
			DurationMs:           float64(duration),
			Explain:              explainResult,
			EstimationErrorValue: estErrorValue,
			EstimationErrorRatio: estErrorRatio,
			RiskOperatorsCount:   riskCount,
		}
		results = append(results, result)
		queryID++
	}

	return results
}

// parseExplainAnalyze parses EXPLAIN ANALYZE output
func (dbm *DBManager) parseExplainAnalyze(explainText string) (float64, float64, int) {
	// Simplified parsing - in real implementation would parse the actual explain output
	lines := strings.Split(explainText, "\n")
	operators := []map[string]interface{}{}

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "(") && strings.HasSuffix(line, ")") {
			// Parse operator info (simplified)
			operator := map[string]interface{}{
				"est_rows": 100.0,
				"act_rows": 110.0,
			}

			estRows := operator["est_rows"].(float64)
			actRows := operator["act_rows"].(float64)

			estErrorValue := estRows - actRows
			if estErrorValue < 0 {
				estErrorValue = -estErrorValue
			}

			estErrorRatio := maxFloat64(estRows, actRows) / minFloat64(estRows, actRows)
			isRisk := estErrorRatio >= 10 && estErrorValue >= 1000

			operator["estimation_error_value"] = estErrorValue
			operator["estimation_error_ratio"] = estErrorRatio
			operator["is_risk"] = isRisk

			operators = append(operators, operator)
		}
	}

	if len(operators) == 0 {
		return 0.0, 0.0, 0
	}

	var totalErrorValue, totalErrorRatio float64
	var riskCount int

	for _, op := range operators {
		totalErrorValue += op["estimation_error_value"].(float64)
		totalErrorRatio += op["estimation_error_ratio"].(float64)
		if op["is_risk"].(bool) {
			riskCount++
		}
	}

	avgErrorValue := totalErrorValue / float64(len(operators))
	avgErrorRatio := totalErrorRatio / float64(len(operators))

	return avgErrorValue, avgErrorRatio, riskCount
}

// GetTableStats gets statistics for specified columns
func (dbm *DBManager) GetTableStats(tableName string, columns []string) map[string]map[string]interface{} {
	dbm.EnsureConnection()
	stats := make(map[string]map[string]interface{})

	for _, col := range columns {
		stats[col] = map[string]interface{}{
			"min": nil,
			"max": nil,
		}

		query := fmt.Sprintf("SELECT MIN(`%s`), MAX(`%s`) FROM `%s`", col, col, tableName)
		var minVal, maxVal sql.NullString

		err := dbm.db.QueryRow(query).Scan(&minVal, &maxVal)
		if err != nil {
			fmt.Printf("      [Warning] Failed to fetch stats for %s.%s: %v\n", tableName, col, err)
			continue
		}

		if minVal.Valid {
			stats[col]["min"] = minVal.String
		}
		if maxVal.Valid {
			stats[col]["max"] = maxVal.String
		}
	}

	return stats
}

// GetStatsHealthy gets stats healthy information for all tables
func (dbm *DBManager) GetStatsHealthy() map[string]float64 {
	dbm.EnsureConnection()
	statsHealthy := make(map[string]float64)

	rows, err := dbm.db.Query("SHOW STATS_HEALTHY")
	if err != nil {
		fmt.Printf("Warning: Could not execute SHOW STATS_HEALTHY: %v\n", err)
		return statsHealthy
	}
	defer rows.Close()

	var dbName, tableName string
	var healthy, healthyRatio float64

	for rows.Next() {
		err := rows.Scan(&dbName, &tableName, &healthy, &healthyRatio)
		if err != nil {
			continue
		}

		// Convert healthy to ratio (assuming healthy is percentage)
		if healthy > 0 {
			statsHealthy[tableName] = healthy / 100.0
		} else {
			statsHealthy[tableName] = 1.0
		}
	}

	return statsHealthy
}

// Close closes the database connection
func (dbm *DBManager) Close() {
	if dbm.db != nil {
		dbm.db.Close()
	}
}

// Helper functions
func maxFloat64(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func minFloat64(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
