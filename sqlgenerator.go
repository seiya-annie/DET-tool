package main

import (
	"fmt"
	"os"
	"strings"
	"time"
)

// SqlGenerator generates SQL statements
type SqlGenerator struct {
	statements []string
}

// NewSqlGenerator creates a new SqlGenerator
func NewSqlGenerator() *SqlGenerator {
	return &SqlGenerator{
		statements: make([]string, 0),
	}
}

// LogDeleteLimit logs a DELETE statement with LIMIT
func (sg *SqlGenerator) LogDeleteLimit(tableName string, limitCount int) {
	if limitCount > 0 {
		sg.statements = append(sg.statements, fmt.Sprintf("DELETE FROM `%s` LIMIT %d;", tableName, limitCount))
	}
}

// LogUpdate logs UPDATE statements for multiple rows
func (sg *SqlGenerator) LogUpdate(tableName string, idColumn string, ids []interface{}, columnNames []string, values [][]interface{}) {
	for i, id := range ids {
		if i >= len(values) {
			break
		}

		sets := []string{}
		for j, colName := range columnNames {
			if j < len(values[i]) {
				sets = append(sets, fmt.Sprintf("`%s`=%s", colName, sg.formatValue(values[i][j])))
			}
		}

		if len(sets) > 0 {
			sql := fmt.Sprintf("UPDATE `%s` SET %s WHERE `%s`=%s;",
				tableName, strings.Join(sets, ", "), idColumn, sg.formatValue(id))
			sg.statements = append(sg.statements, sql)
		}
	}
}

// LogInsert logs INSERT statements for a DataFrame
func (sg *SqlGenerator) LogInsert(tableName string, df *DataFrame) {
	if len(df.columns) == 0 || len(df.data) == 0 {
		return
	}

	columns := strings.Join(df.columns, ", ")

	for _, row := range df.data {
		if len(row) != len(df.columns) {
			continue
		}

		values := make([]string, len(row))
		for i, val := range row {
			values[i] = sg.formatValue(val)
		}

		sql := fmt.Sprintf("INSERT INTO `%s` (%s) VALUES (%s);",
			tableName, columns, strings.Join(values, ", "))
		sg.statements = append(sg.statements, sql)
	}
}

// LogInsertBatch logs INSERT statements in batch for better performance
func (sg *SqlGenerator) LogInsertBatch(tableName string, df *DataFrame, batchSize int) {
	if len(df.columns) == 0 || len(df.data) == 0 {
		return
	}

	if batchSize <= 0 {
		batchSize = 1000 // Default batch size
	}

	columns := strings.Join(df.columns, ", ")

	for i := 0; i < len(df.data); i += batchSize {
		end := i + batchSize
		if end > len(df.data) {
			end = len(df.data)
		}

		valueGroups := []string{}
		for j := i; j < end; j++ {
			row := df.data[j]
			if len(row) != len(df.columns) {
				continue
			}

			values := make([]string, len(row))
			for k, val := range row {
				values[k] = sg.formatValue(val)
			}
			valueGroups = append(valueGroups, fmt.Sprintf("(%s)", strings.Join(values, ", ")))
		}

		if len(valueGroups) > 0 {
			sql := fmt.Sprintf("INSERT INTO `%s` (%s) VALUES %s;",
				tableName, columns, strings.Join(valueGroups, ", "))
			sg.statements = append(sg.statements, sql)
		}
	}
}

// LogCreateTable logs CREATE TABLE statement based on DataFrame
func (sg *SqlGenerator) LogCreateTable(tableName string, df *DataFrame) {
	if len(df.columns) == 0 {
		return
	}

	columns := []string{"`id` bigint NOT NULL AUTO_INCREMENT"}
	indexes := []string{}

	for _, colName := range df.columns {
		sqlType := sg.inferSQLType(df, colName)
		columns = append(columns, fmt.Sprintf("`%s` %s", colName, sqlType))
		indexes = append(indexes, fmt.Sprintf("KEY `idx_%s` (`%s`)", colName, colName))
	}

	createSQL := fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
		%s,
		%s,
		PRIMARY KEY (`+"`id`"+`)
	) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin`,
		tableName,
		strings.Join(columns, ", "),
		strings.Join(indexes, ", "))

	sg.statements = append(sg.statements, fmt.Sprintf("DROP TABLE IF EXISTS %s", tableName))
	sg.statements = append(sg.statements, createSQL)
}

// LogCreateIndex logs CREATE INDEX statements
func (sg *SqlGenerator) LogCreateIndex(tableName string, columns []string, indexType string) {
	if len(columns) == 0 {
		return
	}

	indexName := fmt.Sprintf("idx_%s_%s", tableName, strings.Join(columns, "_"))
	columnList := strings.Join(columns, ", ")

	var sql string
	switch strings.ToUpper(indexType) {
	case "UNIQUE":
		sql = fmt.Sprintf("CREATE UNIQUE INDEX `%s` ON `%s` (%s)", indexName, tableName, columnList)
	case "FULLTEXT":
		sql = fmt.Sprintf("CREATE FULLTEXT INDEX `%s` ON `%s` (%s)", indexName, tableName, columnList)
	default:
		sql = fmt.Sprintf("CREATE INDEX `%s` ON `%s` (%s)", indexName, tableName, columnList)
	}

	sg.statements = append(sg.statements, sql)
}

// LogDropTable logs DROP TABLE statement
func (sg *SqlGenerator) LogDropTable(tableName string) {
	sg.statements = append(sg.statements, fmt.Sprintf("DROP TABLE IF EXISTS `%s`", tableName))
}

// LogTruncateTable logs TRUNCATE TABLE statement
func (sg *SqlGenerator) LogTruncateTable(tableName string) {
	sg.statements = append(sg.statements, fmt.Sprintf("TRUNCATE TABLE `%s`", tableName))
}

// LogAnalyzeTable logs ANALYZE TABLE statement
func (sg *SqlGenerator) LogAnalyzeTable(tableName string) {
	sg.statements = append(sg.statements, fmt.Sprintf("ANALYZE TABLE `%s` ALL COLUMNS", tableName))
}

// LogSetVariable logs SET variable statement
func (sg *SqlGenerator) LogSetVariable(variable string, value interface{}) {
	sg.statements = append(sg.statements, fmt.Sprintf("SET %s = %s", variable, sg.formatValue(value)))
}

// LogUseDatabase logs USE database statement
func (sg *SqlGenerator) LogUseDatabase(databaseName string) {
	sg.statements = append(sg.statements, fmt.Sprintf("USE `%s`", databaseName))
}

// LogComment logs a comment
func (sg *SqlGenerator) LogComment(comment string) {
	lines := strings.Split(comment, "\n")
	for _, line := range lines {
		sg.statements = append(sg.statements, fmt.Sprintf("-- %s", strings.TrimSpace(line)))
	}
}

// LogEmptyLine logs an empty line
func (sg *SqlGenerator) LogEmptyLine() {
	sg.statements = append(sg.statements, "")
}

// Save saves all statements to a file
func (sg *SqlGenerator) Save(filename string) error {
	if len(sg.statements) == 0 {
		return nil
	}

	content := strings.Join(sg.statements, "\n")
	return os.WriteFile(filename, []byte(content), 0644)
}

// GetStatements returns all generated statements
func (sg *SqlGenerator) GetStatements() []string {
	return sg.statements
}

// Clear clears all generated statements
func (sg *SqlGenerator) Clear() {
	sg.statements = make([]string, 0)
}

// Size returns the number of generated statements
func (sg *SqlGenerator) Size() int {
	return len(sg.statements)
}

// formatValue formats a value for SQL
func (sg *SqlGenerator) formatValue(val interface{}) string {
	if val == nil {
		return "NULL"
	}

	switch v := val.(type) {
	case string:
		// Escape single quotes and wrap in quotes
		escaped := strings.ReplaceAll(v, "'", "''")
		return fmt.Sprintf("'%s'", escaped)
	case time.Time:
		return fmt.Sprintf("'%s'", v.Format("2006-01-02 15:04:05"))
	case bool:
		if v {
			return "1"
		}
		return "0"
	case int, int64, int32, int16, int8:
		return fmt.Sprintf("%d", v)
	case float64, float32:
		return fmt.Sprintf("%.6f", v)
	default:
		return fmt.Sprintf("'%v'", v)
	}
}

// inferSQLType infers SQL type from DataFrame column
func (sg *SqlGenerator) inferSQLType(df *DataFrame, colName string) string {
	col := df.GetColumn(colName)
	if len(col) == 0 {
		return "VARCHAR(255)"
	}

	// Check first few non-null values
	sampleSize := minInt(10, len(col))
	intCount := 0
	floatCount := 0
	dateCount := 0
	stringCount := 0

	for i := 0; i < sampleSize; i++ {
		if col[i] == nil {
			continue
		}

		switch col[i].(type) {
		case int, int64, int32, int16, int8:
			intCount++
		case float64, float32:
			floatCount++
		case time.Time:
			dateCount++
		default:
			stringCount++
		}
	}

	// Determine type based on majority
	total := intCount + floatCount + dateCount + stringCount
	if total == 0 {
		return "VARCHAR(255)"
	}

	if floatCount > total/2 {
		return "DOUBLE"
	} else if intCount > total/2 {
		return "BIGINT"
	} else if dateCount > total/2 {
		return "DATETIME"
	} else {
		return "VARCHAR(255)"
	}
}

// Helper functions

// GenerateInsertFromMap generates INSERT statement from map
func (sg *SqlGenerator) GenerateInsertFromMap(tableName string, data map[string]interface{}) string {
	if len(data) == 0 {
		return ""
	}

	columns := make([]string, 0, len(data))
	values := make([]string, 0, len(data))

	for col, val := range data {
		columns = append(columns, fmt.Sprintf("`%s`", col))
		values = append(values, sg.formatValue(val))
	}

	return fmt.Sprintf("INSERT INTO `%s` (%s) VALUES (%s)",
		tableName, strings.Join(columns, ", "), strings.Join(values, ", "))
}

// GenerateUpdateFromMap generates UPDATE statement from map
func (sg *SqlGenerator) GenerateUpdateFromMap(tableName string, data map[string]interface{}, whereClause string) string {
	if len(data) == 0 {
		return ""
	}

	sets := make([]string, 0, len(data))
	for col, val := range data {
		sets = append(sets, fmt.Sprintf("`%s`=%s", col, sg.formatValue(val)))
	}

	return fmt.Sprintf("UPDATE `%s` SET %s WHERE %s",
		tableName, strings.Join(sets, ", "), whereClause)
}

// GenerateSelect generates SELECT statement
func (sg *SqlGenerator) GenerateSelect(tableName string, columns []string, whereClause string, orderBy string, limit int) string {
	var columnList string
	if len(columns) == 0 {
		columnList = "*"
	} else {
		quotedColumns := make([]string, len(columns))
		for i, col := range columns {
			quotedColumns[i] = fmt.Sprintf("`%s`", col)
		}
		columnList = strings.Join(quotedColumns, ", ")
	}

	sql := fmt.Sprintf("SELECT %s FROM `%s`", columnList, tableName)

	if whereClause != "" {
		sql += fmt.Sprintf(" WHERE %s", whereClause)
	}

	if orderBy != "" {
		sql += fmt.Sprintf(" ORDER BY %s", orderBy)
	}

	if limit > 0 {
		sql += fmt.Sprintf(" LIMIT %d", limit)
	}

	return sql
}

// GenerateDelete generates DELETE statement
func (sg *SqlGenerator) GenerateDelete(tableName string, whereClause string) string {
	sql := fmt.Sprintf("DELETE FROM `%s`", tableName)
	if whereClause != "" {
		sql += fmt.Sprintf(" WHERE %s", whereClause)
	}
	return sql
}
