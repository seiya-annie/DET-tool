package main

import (
	"database/sql"
	"encoding/csv"
	"fmt"
	"log"
	"math" // [修复] 添加 math 包
	"os"
	"path/filepath"
	"strconv" // [修复] 添加 strconv 包
	"strings"
	"time"

	mysql "github.com/go-sql-driver/mysql"
)

// DBManager manages database operations
type DBManager struct {
    config DBConfig
    db     *sql.DB
    analyzeMaxRetries int
    analyzeInterval   time.Duration
}

// NewDBManager creates a new DBManager instance
func NewDBManager(config DBConfig) *DBManager {
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

	// Build the column list string
	quotedCols := make([]string, len(header))
	for i, col := range header {
		quotedCols[i] = fmt.Sprintf("`%s`", strings.TrimSpace(col))
	}
	columnListSql := fmt.Sprintf("(%s)", strings.Join(quotedCols, ", "))

	// Replace backslashes for Windows
	absPath = strings.ReplaceAll(absPath, "\\", "/")
	mysql.RegisterLocalFile(absPath)

	// 2. Construct SQL
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
    var partition sql.NullString
    var healthy int

    if rows.Next() {
        err := rows.Scan(&dbNameResult, &tableNameResult, &partition, &healthy)
        if err == nil {
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
        err := rows.Scan(&dbNameResult, &tableNameResult, &partition, &healthy)
        if err == nil {
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

    // Switch to target DB
    if _, err := dbm.db.Exec(fmt.Sprintf("USE `%s`", dbName)); err != nil {
        fmt.Printf("    [Error] Failed to switch to DB %s: %v\n", dbName, err)
        return
    }

    // List all tables
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

        // Wait up to N seconds for healthy=100
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

    // Switch back to default DB
    _, _ = dbm.db.Exec(fmt.Sprintf("USE `%s`", dbm.config.DBName))
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

	// 临时调大当前 Session 的内存限制
	_, err = dbm.db.Exec("SET tidb_mem_quota_query = 2 * 1024 * 1024 * 1024")
	if err != nil {
		fmt.Printf("    [Warning] Failed to increase memory quota: %v\n", err)
	}

	statements := strings.Split(string(content), ";")

	// 分批事务控制
	batchSize := 100 // 每 100 条 SQL 提交一次
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

		// 如果没有开启事务，开启一个新的
		if tx == nil {
			var err error
			tx, err = dbm.db.Begin()
			if err != nil {
				log.Printf("Error starting transaction: %v", err)
				return
			}
		}

		// 执行 SQL
		_, err := tx.Exec(statement)
		if err != nil {
			fmt.Printf("      SQL Error: %v\n      Statement partial: %s\n", err, truncateString(statement, 100))
			// 遇到错误回滚当前批次并退出，或者选择继续
			tx.Rollback()
			tx = nil
			continue
		}

		stmtCount++
		totalCount++

		// 达到批次大小，提交事务
		if stmtCount >= batchSize {
			commitTx()
			stmtCount = 0
			fmt.Printf("\r      Executed %d statements...", totalCount)
		}
	}

	// 提交剩余的事务
	commitTx()
	fmt.Println()
	fmt.Printf("    [DB] Finished executing %d statements.\n", totalCount)
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

		// 1. Execute query to get Duration
		start := time.Now()
		rows, err := dbm.db.Query(query)
		if err != nil {
			fmt.Printf("      Q%d Error: %v\n", queryID, err)
			queryID++
			continue
		}
		rows.Close()
		duration := time.Since(start).Milliseconds()

		// 2. Run EXPLAIN ANALYZE
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
		// Find indices
		for i, col := range columns {
			valuePtrs[i] = &values[i]
			if strings.EqualFold(col, "estRows") {
				estRowIdx = i
			}
			if strings.EqualFold(col, "actRows") {
				actRowIdx = i
			}
		}

		// Write Header
		for i, col := range columns {
			sb.WriteString(col)
			if i < count-1 {
				sb.WriteString("\t")
			}
		}
		sb.WriteString("\n")

		for explainRows.Next() {
			err := explainRows.Scan(valuePtrs...)
			if err != nil {
				continue
			}

			// Build Explain String
			for i, val := range values {
				var v interface{}
				b, ok := val.([]byte)
				if ok {
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

			// Calculate Error
			if estRowIdx != -1 && actRowIdx != -1 {
				est := dbm.toFloat(values[estRowIdx])
				act := dbm.toFloat(values[actRowIdx])

				act = math.Max(1.0, act)
				est = math.Max(1.0, est)

				errVal := math.Abs(act - est)
				errRatio := math.Max(act, est) / math.Min(act, est)

				// [修复点 1]：无论是否是 Risk SQL，都记录当前查询中最大的误差值
				if errRatio > maxErrorRatio {
					maxErrorRatio = errRatio
					maxErrorValue = errVal
				}

				// [修复点 2]：只有满足阈值时，才计入 Risk Count
				if errRatio >= 10 && errVal >= 1000 {
					riskCount++
				}
			}
		}
		explainRows.Close()

        result := QueryResult{
            QueryID:              queryID,
            Query:                query,
            QueryLabel:           extractQueryLabel(query),
            DurationMs:           float64(duration),
            Explain:              sb.String(), // 包含完整换行的字符串
            EstimationErrorValue: maxErrorValue,
            EstimationErrorRatio: maxErrorRatio,
            RiskOperatorsCount:   riskCount,
        }
		results = append(results, result)
		queryID++
	}

	return results
}

// ExecuteAndExplainQueriesOnDB executes a list of raw SQL queries on the given database name
// and returns QueryResult entries including EXPLAIN ANALYZE information.
func (dbm *DBManager) ExecuteAndExplainQueriesOnDB(targetDB string, queries []string) []QueryResult {
    results := []QueryResult{}

    if targetDB == "" {
        return results
    }

    // Switch to target DB
    _, err := dbm.db.Exec(fmt.Sprintf("USE `%s`", targetDB))
    if err != nil {
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

        res := QueryResult{
            QueryID:              queryID,
            Query:                q,
            QueryLabel:           extractQueryLabel(q),
            DurationMs:           float64(duration),
            Explain:              sb.String(),
            EstimationErrorValue: maxErrorValue,
            EstimationErrorRatio: maxErrorRatio,
            RiskOperatorsCount:   riskCount,
        }
        results = append(results, res)
        queryID++
    }

    // Switch back to original DB for safety
    _, _ = dbm.db.Exec(fmt.Sprintf("USE `%s`", dbm.config.DBName))

    return results
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

//// Helper to extract float value from string like "key:value"
//func (dbm *DBManager) extractValue(line, key string) float64 {
//	keyStr := key + ":"
//	idx := strings.Index(line, keyStr)
//	if idx == -1 {
//		return 0.0
//	}
//
//	start := idx + len(keyStr)
//	end := start
//	for end < len(line) {
//		c := line[end]
//		// consume number chars
//		if (c >= '0' && c <= '9') || c == '.' {
//			end++
//		} else {
//			break
//		}
//	}
//
//	if start == end {
//		return 0.0
//	}
//
//	valStr := line[start:end]
//	val, err := strconv.ParseFloat(valStr, 64)
//	if err != nil {
//		return 0.0
//	}
//	return val
//}

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
// Implements Requirement 1: Return map[string]int based on SHOW STATS_HEALTHY
func (dbm *DBManager) GetStatsHealthy() map[string]int {
	dbm.EnsureConnection()
	statsHealthy := make(map[string]int)

	rows, err := dbm.db.Query("SHOW STATS_HEALTHY")
	if err != nil {
		fmt.Printf("Warning: Could not execute SHOW STATS_HEALTHY: %v\n", err)
		return statsHealthy
	}
	defer rows.Close()

	// Output format: Db_name | Table_name | Partition_name | Healthy
	var dbName, tableName string
	var healthy int

	for rows.Next() {
		// [修复] 移除未使用的变量，直接 scan 到 sql.NullString 处理可能为 NULL 的 partition
		var partName sql.NullString
		err := rows.Scan(&dbName, &tableName, &partName, &healthy)
		if err != nil {
			continue
		}

		// Store healthy value (0-100)
		statsHealthy[tableName] = healthy
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
