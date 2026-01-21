package db

import (
	"database/sql"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	itypes "det-tool/internal/types"

	mysql "github.com/go-sql-driver/mysql"
)

// DBManager manages database operations
type DBManager struct {
	config            itypes.DBConfig
	db                *sql.DB
	analyzeMaxRetries int
	analyzeInterval   time.Duration
	dmlBatchSize      int
}

// NewDBManager creates a new DBManager instance
func NewDBManager(config itypes.DBConfig) *DBManager {
	dbManager := &DBManager{config: config}
	dbManager.connect()
	return dbManager
}

// SetAnalyzeWaitPolicy sets waiting policy for stats healthy after ANALYZE
func (dbm *DBManager) SetAnalyzeWaitPolicy(maxRetries int, interval time.Duration) {
	if maxRetries > 0 {
		dbm.analyzeMaxRetries = maxRetries
	}
	if interval > 0 {
		dbm.analyzeInterval = interval
	}
}

func (dbm *DBManager) getAnalyzeParams() (int, time.Duration) {
	retries := dbm.analyzeMaxRetries
	if retries <= 0 {
		retries = 20
	}
	interval := dbm.analyzeInterval
	if interval <= 0 {
		interval = 1 * time.Second
	}
	return retries, interval
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

// SetDMLBatchSize sets the per-transaction statement batch size when executing SQL scripts
func (dbm *DBManager) SetDMLBatchSize(n int) {
	if n > 0 {
		dbm.dmlBatchSize = n
	}
}

// InitDB initializes the database
func (dbm *DBManager) InitDB(dropIfExists bool) {
	dbName := dbm.config.DBName
	if dropIfExists {
		_, err := dbm.db.Exec(fmt.Sprintf("DROP DATABASE IF EXISTS %s", dbName))
		if err != nil {
			log.Printf("Error dropping database: %v", err)
		}
		_, err = dbm.db.Exec(fmt.Sprintf("CREATE DATABASE IF NOT EXISTS %s", dbName))
		if err != nil {
			log.Printf("Error creating database: %v", err)
		}
	}
	if _, err := dbm.db.Exec(fmt.Sprintf("USE %s", dbName)); err != nil {
		log.Printf("Error selecting database: %v", err)
	}
	fmt.Printf(">>> Selected Database: %s\n", dbName)
}

// TableExists checks whether a table exists in the current database.
func (dbm *DBManager) TableExists(tableName string) bool {
	dbm.EnsureConnection()
	q := fmt.Sprintf("SELECT 1 FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = '%s' LIMIT 1", tableName)
	var x int
	if err := dbm.db.QueryRow(q).Scan(&x); err != nil {
		return false
	}
	return x == 1
}

// DisableAutoAnalyze disables TiDB auto analyze
func (dbm *DBManager) DisableAutoAnalyze() {
	fmt.Println("    [DB] Disabling Global Auto Analyze...")
	_, err := dbm.db.Exec("SET GLOBAL tidb_enable_auto_analyze = OFF")
	if err != nil {
		fmt.Printf("    [Warning] Failed to disable auto analyze: %v\n", err)
	}
}

// Frame is a minimal interface implemented by DataFrame types
type Frame interface {
	Columns() []string
	GetColumn(name string) []interface{}
}

// CreateTable creates a table based on DataFrame structure. Optional partitionClause
// is appended verbatim to the DDL (e.g., PARTITION BY RANGE ...).
func (dbm *DBManager) CreateTable(tableName string, df Frame, partitionClause string) {
	cols := []string{}
	indexes := []string{}
	for _, colName := range df.Columns() {
		sqlType := dbm.inferSQLType(df, colName)
		cols = append(cols, fmt.Sprintf("`%s` %s", colName, sqlType))
		indexes = append(indexes, fmt.Sprintf("KEY `idx_%s` (`%s`)", colName, colName))
	}
	cols = append([]string{"`id` bigint NOT NULL AUTO_INCREMENT"}, cols...)
	pkCols := []string{"`id`"}
	if pkProvider, ok := df.(interface{ PrimaryKeys() []string }); ok {
		if pks := pkProvider.PrimaryKeys(); len(pks) > 0 {
			pkCols = pkCols[:0]
			for _, c := range pks {
				pkCols = append(pkCols, fmt.Sprintf("`%s`", c))
			}
		}
	}
	ddl := fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
        %s,
        %s,
        PRIMARY KEY (%s)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin %s`,
		tableName, strings.Join(cols, ", "), strings.Join(indexes, ", "), strings.Join(pkCols, ", "), strings.TrimSpace(partitionClause))
	fmt.Println(ddl)
	_, err := dbm.db.Exec(fmt.Sprintf("DROP TABLE IF EXISTS %s", tableName))
	if err != nil {
		log.Printf("Error dropping table: %v", err)
	}
	_, err = dbm.db.Exec(ddl)
	if err != nil {
		log.Printf("Error creating table: %v", err)
	}
	fmt.Printf("    [DB] Table created: %s\n", tableName)
}

// inferSQLType infers SQL type from DataFrame column data
func (dbm *DBManager) inferSQLType(df Frame, colName string) string {
	col := df.GetColumn(colName)
	if len(col) == 0 {
		return "VARCHAR(255)"
	}
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

	f, err := os.Open(absPath)
	if err != nil {
		log.Printf("    [Error] Failed to open CSV to read header: %v", err)
		return
	}
	csvReader := csv.NewReader(f)
	header, err := csvReader.Read()
	f.Close()
	if err != nil {
		log.Printf("    [Error] Failed to read CSV header: %v", err)
		return
	}

	quotedCols := make([]string, len(header))
	for i, col := range header {
		quotedCols[i] = fmt.Sprintf("`%s`", strings.TrimSpace(col))
	}
	columnListSql := fmt.Sprintf("(%s)", strings.Join(quotedCols, ", "))
	absPath = strings.ReplaceAll(absPath, "\\", "/")
	mysql.RegisterLocalFile(absPath)
	sql := fmt.Sprintf(`LOAD DATA LOCAL INFILE '%s' INTO TABLE %s 
        FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' 
        LINES TERMINATED BY '\n' 
        IGNORE 1 LINES 
        %s`, absPath, tableName, columnListSql)
	_, _ = dbm.db.Exec("SET GLOBAL local_infile = 1")
	if _, err := dbm.db.Exec(sql); err != nil {
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
	var partition sql.NullString
	var healthy int
	if rows.Next() {
		if err := rows.Scan(&dbNameResult, &tableNameResult, &partition, &healthy); err == nil {
			return healthy
		}
	}
	return 0
}

// GetSingleTableHealthInDB gets stats health for a specific table in a specific DB
func (dbm *DBManager) GetSingleTableHealthInDB(dbName, tableName string) int {
	query := fmt.Sprintf("SHOW STATS_HEALTHY WHERE Db_name = '%s' AND Table_name = '%s'", dbName, tableName)
	rows, err := dbm.db.Query(query)
	if err != nil {
		return 0
	}
	defer rows.Close()
	var dbNameResult, tableNameResult string
	var partition sql.NullString
	var healthy int
	if rows.Next() {
		if err := rows.Scan(&dbNameResult, &tableNameResult, &partition, &healthy); err == nil {
			return healthy
		}
	}
	return 0
}

// AnalyzeAllTablesInDB analyzes all tables in the specified database
func (dbm *DBManager) AnalyzeAllTablesInDB(dbName string) {
	if strings.TrimSpace(dbName) == "" {
		return
	}
	fmt.Printf("    [DB] Analyzing all tables in database: %s\n", dbName)
	if _, err := dbm.db.Exec(fmt.Sprintf("USE `%s`", dbName)); err != nil {
		fmt.Printf("    [Error] Failed to switch to DB %s: %v\n", dbName, err)
		return
	}
	tblRows, err := dbm.db.Query("SHOW TABLES")
	if err != nil {
		fmt.Printf("    [Error] Failed to list tables in %s: %v\n", dbName, err)
		return
	}
	defer tblRows.Close()
	var tables []string
	for tblRows.Next() {
		var t string
		if err := tblRows.Scan(&t); err == nil {
			tables = append(tables, t)
		}
	}
	maxRetries, interval := dbm.getAnalyzeParams()
	for _, t := range tables {
		fmt.Printf("    [DB] ANALYZE TABLE %s.%s ALL COLUMNS ...\n", dbName, t)
		startTime := time.Now()
		if _, err := dbm.db.Exec(fmt.Sprintf("ANALYZE TABLE `%s` ALL COLUMNS", t)); err != nil {
			fmt.Printf("      [Error] Analyze failed for %s.%s: %v\n", dbName, t, err)
			continue
		}
		for i := 0; i < maxRetries; i++ {
			health := dbm.GetSingleTableHealthInDB(dbName, t)
			if health == 100 {
				break
			}
			time.Sleep(interval)
			if i == maxRetries-1 {
				fmt.Printf("      [Warning] Stats health for %s.%s reached %d%%, timeout.\n", dbName, t, health)
			}
		}
		duration := time.Since(startTime)
		finalHealth := dbm.GetSingleTableHealthInDB(dbName, t)
		fmt.Printf("      [DB] Analyze finished for %s.%s in %.2fs (Health: %d%%)\n", dbName, t, duration.Seconds(), finalHealth)
	}
	_, _ = dbm.db.Exec(fmt.Sprintf("USE `%s`", dbm.config.DBName))
}

// AnalyzeTable analyzes a table and waits for health to reach 100
func (dbm *DBManager) AnalyzeTable(tableName string) {
	fmt.Printf("    [DB] Executing Manual Analyze: ANALYZE TABLE %s ALL COLUMNS ...\n", tableName)
	startTime := time.Now()
	if _, err := dbm.db.Exec(fmt.Sprintf("ANALYZE TABLE %s ALL COLUMNS", tableName)); err != nil {
		fmt.Printf("    [Error] Analyze failed: %v\n", err)
		return
	}
	fmt.Println("    [DB] Waiting for stats to become healthy (100%)...")
	maxRetries, interval := dbm.getAnalyzeParams()
	for i := 0; i < maxRetries; i++ {
		health := dbm.GetSingleTableHealth(tableName)
		if health == 100 {
			break
		}
		time.Sleep(interval)
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
	_, err = dbm.db.Exec("SET tidb_mem_quota_query = 2 * 1024 * 1024 * 1024")
	if err != nil {
		fmt.Printf("    [Warning] Failed to increase memory quota: %v\n", err)
	}
	_, _ = dbm.db.Exec("SET GLOBAL local_infile = 1")
	statements := strings.Split(string(content), ";")
	batchSize := dbm.dmlBatchSize
	if batchSize <= 0 {
		batchSize = 100
	}
	var tx *sql.Tx
	stmtCount := 0
	totalCount := 0
	commitTx := func() {
		if tx != nil {
			if err := tx.Commit(); err != nil {
				log.Printf("Error committing transaction: %v", err)
			}
			tx = nil
		}
	}
	for _, statement := range statements {
		statement = strings.TrimSpace(statement)
		if statement == "" {
			continue
		}
		if tx == nil {
			var err error
			tx, err = dbm.db.Begin()
			if err != nil {
				log.Printf("Error starting transaction: %v", err)
				return
			}
		}
		up := strings.ToUpper(statement)
		if strings.Contains(up, "LOAD DATA LOCAL INFILE") {
			if i := strings.Index(up, "INFILE"); i >= 0 {
				rest := statement[i:]
				if j := strings.IndexByte(rest, '\''); j >= 0 {
					rest2 := rest[j+1:]
					if k := strings.IndexByte(rest2, '\''); k >= 0 {
						mysql.RegisterLocalFile(rest2[:k])
					}
				}
			}
		}
		if _, err := tx.Exec(statement); err != nil {
			fmt.Printf("      SQL Error: %v\n      Statement partial: %s\n", err, truncateString(statement, 100))
			tx.Rollback()
			tx = nil
			continue
		}
		stmtCount++
		totalCount++
		if stmtCount >= batchSize {
			commitTx()
			stmtCount = 0
			fmt.Printf("\r      Executed %d statements...", totalCount)
		}
	}
	commitTx()
	fmt.Println()
	fmt.Printf("    [DB] Finished executing %d statements.\n", totalCount)
}

// ExecuteAndExplain executes queries and returns results with explain plans
func (dbm *DBManager) ExecuteAndExplain(queryFile string) []itypes.QueryResult {
	if _, err := os.Stat(queryFile); os.IsNotExist(err) {
		return []itypes.QueryResult{}
	}
	content, err := os.ReadFile(queryFile)
	if err != nil {
		log.Printf("Error reading query file: %v", err)
		return []itypes.QueryResult{}
	}
	queries := strings.Split(string(content), ";")
	results := []itypes.QueryResult{}
	queryID := 1
	for _, query := range queries {
		query = strings.TrimSpace(query)
		if query == "" || strings.HasPrefix(query, "--") {
			continue
		}
		start := time.Now()
		rows, err := dbm.db.Query(query)
		if err != nil {
			fmt.Printf("      Q%d Error: %v\n", queryID, err)
			queryID++
			continue
		}
		rows.Close()
		duration := time.Since(start).Milliseconds()
		explainQuery := fmt.Sprintf("EXPLAIN ANALYZE %s", query)
		explainRows, err := dbm.db.Query(explainQuery)
		if err != nil {
			fmt.Printf("      Explain Error: %v\n", err)
			queryID++
			continue
		}
		columns, _ := explainRows.Columns()
		count := len(columns)
		values := make([]interface{}, count)
		valuePtrs := make([]interface{}, count)
		var sb strings.Builder
		maxErrorRatio := 0.0
		maxErrorValue := 0.0
		riskCount := 0
		estRowIdx := -1
		actRowIdx := -1
		for i, col := range columns {
			valuePtrs[i] = &values[i]
			if strings.EqualFold(col, "estRows") {
				estRowIdx = i
			}
			if strings.EqualFold(col, "actRows") {
				actRowIdx = i
			}
		}
		for i, col := range columns {
			sb.WriteString(col)
			if i < count-1 {
				sb.WriteString("\t")
			}
		}
		sb.WriteString("\n")
		for explainRows.Next() {
			if err := explainRows.Scan(valuePtrs...); err != nil {
				continue
			}
			for i, val := range values {
				var v interface{}
				if b, ok := val.([]byte); ok {
					v = string(b)
				} else {
					v = val
				}
				sb.WriteString(fmt.Sprintf("%v", v))
				if i < count-1 {
					sb.WriteString("\t")
				}
			}
			sb.WriteString("\n")
			if estRowIdx != -1 && actRowIdx != -1 {
				est := dbm.toFloat(values[estRowIdx])
				act := dbm.toFloat(values[actRowIdx])
				act = math.Max(1.0, act)
				est = math.Max(1.0, est)
				errVal := math.Abs(act - est)
				errRatio := math.Max(act, est) / math.Min(act, est)
				if errRatio > maxErrorRatio {
					maxErrorRatio = errRatio
					maxErrorValue = errVal
				}
				if errRatio >= 10 && errVal >= 1000 {
					riskCount++
				}
			}
		}
		explainRows.Close()
		result := itypes.QueryResult{
			QueryID:              queryID,
			Query:                query,
			QueryLabel:           extractQueryLabel(query),
			DurationMs:           float64(duration),
			Explain:              sb.String(),
			EstimationErrorValue: maxErrorValue,
			EstimationErrorRatio: maxErrorRatio,
			RiskOperatorsCount:   riskCount,
		}
		// If this is a bad case, try to generate plan replayer dump and capture link
		if maxErrorRatio >= 10 && maxErrorValue >= 1000 {
			if link := dbm.dumpPlanReplayer(query); link != "" {
				result.PlanReplayerLink = link
			}
		}
		results = append(results, result)
		queryID++
	}
	return results
}

// ExecuteAndExplainQueriesOnDB executes a list of raw SQL queries on the given database name
func (dbm *DBManager) ExecuteAndExplainQueriesOnDB(targetDB string, queries []string) []itypes.QueryResult {
	results := []itypes.QueryResult{}
	if targetDB == "" {
		return results
	}
	if _, err := dbm.db.Exec(fmt.Sprintf("USE `%s`", targetDB)); err != nil {
		fmt.Printf("      [Warning] Cannot switch to DB %s: %v\n", targetDB, err)
		return results
	}
	queryID := 1
	for _, query := range queries {
		q := strings.TrimSpace(query)
		if q == "" || strings.HasPrefix(q, "--") {
			continue
		}
		start := time.Now()
		rows, err := dbm.db.Query(q)
		if err != nil {
			fmt.Printf("      Q%d Error: %v\n", queryID, err)
			queryID++
			continue
		}
		rows.Close()
		duration := time.Since(start).Milliseconds()
		explainQuery := fmt.Sprintf("EXPLAIN ANALYZE %s", q)
		explainRows, err := dbm.db.Query(explainQuery)
		if err != nil {
			fmt.Printf("      Explain Error: %v\n", err)
			queryID++
			continue
		}
		columns, _ := explainRows.Columns()
		count := len(columns)
		values := make([]interface{}, count)
		valuePtrs := make([]interface{}, count)
		var sb strings.Builder
		maxErrorRatio := 0.0
		maxErrorValue := 0.0
		riskCount := 0
		estRowIdx := -1
		actRowIdx := -1
		for i, col := range columns {
			valuePtrs[i] = &values[i]
			if strings.EqualFold(col, "estRows") {
				estRowIdx = i
			}
			if strings.EqualFold(col, "actRows") {
				actRowIdx = i
			}
		}
		for i, col := range columns {
			sb.WriteString(col)
			if i < count-1 {
				sb.WriteString("\t")
			}
		}
		sb.WriteString("\n")
		for explainRows.Next() {
			if err := explainRows.Scan(valuePtrs...); err != nil {
				continue
			}
			for i, val := range values {
				var v interface{}
				if b, ok := val.([]byte); ok {
					v = string(b)
				} else {
					v = val
				}
				sb.WriteString(fmt.Sprintf("%v", v))
				if i < count-1 {
					sb.WriteString("\t")
				}
			}
			sb.WriteString("\n")
			if estRowIdx != -1 && actRowIdx != -1 {
				est := dbm.toFloat(values[estRowIdx])
				act := dbm.toFloat(values[actRowIdx])
				act = math.Max(1.0, act)
				est = math.Max(1.0, est)
				errVal := math.Abs(act - est)
				errRatio := math.Max(act, est) / math.Min(act, est)
				if errRatio > maxErrorRatio {
					maxErrorRatio = errRatio
					maxErrorValue = errVal
				}
				if errRatio >= 10 && errVal >= 1000 {
					riskCount++
				}
			}
		}
		explainRows.Close()
		res := itypes.QueryResult{
			QueryID:              queryID,
			Query:                q,
			QueryLabel:           extractQueryLabel(q),
			DurationMs:           float64(duration),
			Explain:              sb.String(),
			EstimationErrorValue: maxErrorValue,
			EstimationErrorRatio: maxErrorRatio,
			RiskOperatorsCount:   riskCount,
		}
		if maxErrorRatio >= 10 && maxErrorValue >= 1000 {
			if link := dbm.dumpPlanReplayer(q); link != "" {
				res.PlanReplayerLink = link
			}
		}
		results = append(results, res)
		queryID++
	}
	_, _ = dbm.db.Exec(fmt.Sprintf("USE `%s`", dbm.config.DBName))
	return results
}

// dumpPlanReplayer runs TiDB plan replayer for the given SQL and tries to download the dump file.
// Returns a link (relative path under output/planreplayer or HTTP URL) to be used in reports.
func (dbm *DBManager) dumpPlanReplayer(sqlText string) string {
	// 1) Execute plan replayer dump explain <sql>
	stmt := fmt.Sprintf("plan replayer dump explain %s", sqlText)
	if _, err := dbm.db.Exec(stmt); err != nil {
		fmt.Printf("      [PlanReplayer] dump explain failed: %v\n", err)
		return ""
	}
	// 2) Fetch last token
	var token sql.NullString
	if err := dbm.db.QueryRow("SELECT @@tidb_last_plan_replayer_token").Scan(&token); err != nil || !token.Valid {
		fmt.Printf("      [PlanReplayer] fetch token failed: %v\n", err)
		return ""
	}
	t := strings.TrimSpace(token.String)
	if t == "" {
		return ""
	}
	// 3) Build download URL and try to curl
	statusPort := dbm.config.StatusPort
	if statusPort == 0 {
		statusPort = 10080
	}
	url := fmt.Sprintf("http://%s:%d/plan_replayer/dump/%s", dbm.config.Host, statusPort, t)
	// Ensure output directory exists
	destDir := filepath.Join("output", "planreplayer")
	_ = os.MkdirAll(destDir, 0755)
	// Determine file name; TiDB may already include .zip in token
	fileName := t
	low := strings.ToLower(fileName)
	if !strings.HasSuffix(low, ".zip") {
		fileName = fileName + ".zip"
	}
	destPath := filepath.Join(destDir, fileName)
	cmd := exec.Command("curl", "-sS", "-o", destPath, url)
	if out, err := cmd.CombinedOutput(); err != nil {
		fmt.Printf("      [PlanReplayer] curl download failed: %v, output=%s\n", err, string(out))
		// Fallback to URL if cannot download
		return url
	}
	// Return relative link from report dir if possible: ../planreplayer/<token>.zip
	rel := filepath.ToSlash(filepath.Join("..", "planreplayer", fileName))
	return rel
}

// GetTableStats gets statistics for specified columns
func (dbm *DBManager) GetTableStats(tableName string, columns []string) map[string]map[string]interface{} {
	dbm.EnsureConnection()
	stats := make(map[string]map[string]interface{})
	for _, col := range columns {
		stats[col] = map[string]interface{}{"min": nil, "max": nil}
		query := fmt.Sprintf("SELECT MIN(`%s`), MAX(`%s`) FROM `%s`", col, col, tableName)
		var minVal, maxVal sql.NullString
		if err := dbm.db.QueryRow(query).Scan(&minVal, &maxVal); err != nil {
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
func (dbm *DBManager) GetStatsHealthy() map[string]int {
	dbm.EnsureConnection()
	statsHealthy := make(map[string]int)
	rows, err := dbm.db.Query("SHOW STATS_HEALTHY")
	if err != nil {
		fmt.Printf("Warning: Could not execute SHOW STATS_HEALTHY: %v\n", err)
		return statsHealthy
	}
	defer rows.Close()
	var dbName, tableName string
	var healthy int
	for rows.Next() {
		var partName sql.NullString
		if err := rows.Scan(&dbName, &tableName, &partName, &healthy); err != nil {
			continue
		}
		statsHealthy[tableName] = healthy
	}
	return statsHealthy
}

// GetStatsHealthyForDB returns stats healthy values for all tables in a specific database.
func (dbm *DBManager) GetStatsHealthyForDB(dbName string) map[string]int {
	result := make(map[string]int)
	if strings.TrimSpace(dbName) == "" {
		return result
	}
	dbm.EnsureConnection()
	query := fmt.Sprintf("SHOW STATS_HEALTHY WHERE Db_name = '%s'", dbName)
	rows, err := dbm.db.Query(query)
	if err != nil {
		fmt.Printf("Warning: Could not execute SHOW STATS_HEALTHY for %s: %v\n", dbName, err)
		return result
	}
	defer rows.Close()
	var dbNameResult, tableName string
	var healthy int
	for rows.Next() {
		var partName sql.NullString
		if err := rows.Scan(&dbNameResult, &tableName, &partName, &healthy); err == nil {
			result[tableName] = healthy
		}
	}
	return result
}

// DumpStatsMetaForDB prints SHOW STATS_META rows for a specific database.
// This helps correlate TiDB's modify_count (key-level) with our line-level changes.
func (dbm *DBManager) DumpStatsMetaForDB(dbName string) {
	if strings.TrimSpace(dbName) == "" {
		return
	}
	dbm.EnsureConnection()
	query := fmt.Sprintf("SHOW STATS_META WHERE Db_name = '%s'", dbName)
	rows, err := dbm.db.Query(query)
	if err != nil {
		fmt.Printf("[DB] SHOW STATS_META failed for %s: %v\n", dbName, err)
		return
	}
	defer rows.Close()
	fmt.Printf("\n[DB] STATS_META for DB=%s\n", dbName)
	fmt.Println("Db_name | Table_name | Partition_name | Update_time | Modify_count | Row_count | Last_analyze_time")
	for rows.Next() {
		var dbNameRes, tableName, partition, updateTime, modifyCount, rowCount, lastAnalyze sql.NullString
		if err := rows.Scan(&dbNameRes, &tableName, &partition, &updateTime, &modifyCount, &rowCount, &lastAnalyze); err != nil {
			// Fallback: try fewer columns if TiDB version differs
			var db2, tbl2, part2 sql.NullString
			var upd2, mod2, row2, last2 sql.NullString
			_ = rows.Scan(&db2, &tbl2, &part2, &upd2, &mod2, &row2, &last2)
			dbNameRes, tableName, partition, updateTime, modifyCount, rowCount, lastAnalyze = db2, tbl2, part2, upd2, mod2, row2, last2
		}
		val := func(ns sql.NullString) string {
			if ns.Valid {
				return ns.String
			}
			return "NULL"
		}
		fmt.Printf("%s | %s | %s | %s | %s | %s | %s\n",
			val(dbNameRes), val(tableName), val(partition), val(updateTime), val(modifyCount), val(rowCount), val(lastAnalyze))
	}
}

// GetRandomIDs fetches N random primary keys (id) from a table
func (dbm *DBManager) GetRandomIDs(tableName string, n int) []int64 {
	if n <= 0 {
		return nil
	}
	dbm.EnsureConnection()
	q := fmt.Sprintf("SELECT id FROM `%s` WHERE id IS NOT NULL ORDER BY RAND() LIMIT %d", tableName, n)
	rows, err := dbm.db.Query(q)
	if err != nil {
		return nil
	}
	defer rows.Close()
	res := make([]int64, 0, n)
	for rows.Next() {
		var id sql.NullInt64
		if err := rows.Scan(&id); err == nil && id.Valid {
			res = append(res, id.Int64)
		}
	}
	return res
}

// Close closes the database connection
func (dbm *DBManager) Close() {
	if dbm.db != nil {
		dbm.db.Close()
	}
}

// Helper to safely convert interface to float64
func (dbm *DBManager) toFloat(val interface{}) float64 {
	if val == nil {
		return 0.0
	}
	switch v := val.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int64:
		return float64(v)
	case int:
		return float64(v)
	case []byte:
		f, _ := strconv.ParseFloat(string(v), 64)
		return f
	case string:
		f, _ := strconv.ParseFloat(v, 64)
		return f
	default:
		return 0.0
	}
}

// Local helpers copied to avoid dependency on main package
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	if maxLen <= 3 {
		return s[:maxLen]
	}
	return s[:maxLen-3] + "..."
}

// extractQueryLabel parses SQL and extracts a query label if present.
func extractQueryLabel(sqltxt string) string {
	s := strings.TrimSpace(sqltxt)
	upper := strings.ToUpper(s)
	if strings.HasPrefix(upper, "/*") {
		if end := strings.Index(upper, "*/"); end > 2 {
			head := strings.TrimSpace(s[2:end])
			uhead := strings.ToUpper(head)
			const key = "LABEL:"
			if idx := strings.Index(uhead, key); idx >= 0 {
				label := strings.TrimSpace(head[idx+len(key):])
				label = strings.Trim(label, " -*\t\n")
				return label
			}
		}
	}
	if nl := strings.IndexByte(s, '\n'); nl >= 0 {
		first := strings.TrimSpace(s[:nl])
		ufirst := strings.ToUpper(first)
		if strings.HasPrefix(ufirst, "--") || strings.HasPrefix(ufirst, "#") {
			const key = "LABEL:"
			if idx := strings.Index(ufirst, key); idx >= 0 {
				label := strings.TrimSpace(first[idx+len(key):])
				label = strings.Trim(label, " -*\t\n")
				return label
			}
		}
	}
	if idx := strings.Index(upper, "/* LABEL:"); idx >= 0 {
		rest := s[idx+2:]
		urest := strings.ToUpper(rest)
		if j := strings.Index(urest, "LABEL:"); j >= 0 {
			after := rest[j+len("LABEL:"):]
			if k := strings.Index(after, "*/"); k >= 0 {
				after = after[:k]
			}
			label := strings.TrimSpace(after)
			label = strings.Trim(label, " -*\t\n")
			if label != "" {
				return label
			}
		}
	}
	return ""
}
