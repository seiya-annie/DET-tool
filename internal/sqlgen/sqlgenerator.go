package sqlgen

import (
	"fmt"
	"os"
	"strings"
	"time"
)

// Frame minimal interface used to infer types
type Frame interface {
	GetColumn(name string) []interface{}
}

// SqlGenerator generates SQL statements
type SqlGenerator struct{ statements []string }

func NewSqlGenerator() *SqlGenerator { return &SqlGenerator{statements: make([]string, 0)} }

func (sg *SqlGenerator) LogDeleteLimit(tableName string, limitCount int) {
	if limitCount > 0 {
		sg.statements = append(sg.statements, fmt.Sprintf("DELETE FROM `%s` LIMIT %d;", tableName, limitCount))
	}
}

func (sg *SqlGenerator) LogDeleteByIDs(tableName, idColumn string, ids []interface{}, batchSize int) {
	if len(ids) == 0 {
		return
	}
	if batchSize <= 0 {
		batchSize = 1000
	}
	toSQL := func(v interface{}) string {
		switch x := v.(type) {
		case string:
			return fmt.Sprintf("'%s'", strings.ReplaceAll(x, "'", "''"))
		default:
			return fmt.Sprintf("%v", x)
		}
	}
	for i := 0; i < len(ids); i += batchSize {
		end := i + batchSize
		if end > len(ids) {
			end = len(ids)
		}
		vals := make([]string, end-i)
		for j := i; j < end; j++ {
			vals[j-i] = toSQL(ids[j])
		}
		sg.statements = append(sg.statements, fmt.Sprintf("DELETE FROM `%s` WHERE `%s` IN (%s);", tableName, idColumn, strings.Join(vals, ", ")))
	}
}

func (sg *SqlGenerator) LogUpdate(tableName, idColumn string, ids []interface{}, columnNames []string, values [][]interface{}) {
	for i, id := range ids {
		if i >= len(values) {
			break
		}
		sets := make([]string, 0, len(columnNames))
		for j, colName := range columnNames {
			if j < len(values[i]) {
				sets = append(sets, fmt.Sprintf("`%s`=%s", colName, sg.formatValue(values[i][j])))
			}
		}
		if len(sets) > 0 {
			sg.statements = append(sg.statements, fmt.Sprintf("UPDATE `%s` SET %s WHERE `%s`=%s;", tableName, strings.Join(sets, ", "), idColumn, sg.formatValue(id)))
		}
	}
}

func (sg *SqlGenerator) LogInsertBatch(tableName string, df Frame, rows [][]interface{}, columns []string, batchSize int) {
	if len(columns) == 0 || len(rows) == 0 {
		return
	}
	if batchSize <= 0 {
		batchSize = 1000
	}
	colList := strings.Join(columns, ", ")
	for i := 0; i < len(rows); i += batchSize {
		end := i + batchSize
		if end > len(rows) {
			end = len(rows)
		}
		valueGroups := []string{}
		for j := i; j < end; j++ {
			row := rows[j]
			vals := make([]string, len(row))
			for k, v := range row {
				vals[k] = sg.formatValue(v)
			}
			valueGroups = append(valueGroups, fmt.Sprintf("(%s)", strings.Join(vals, ", ")))
		}
		if len(valueGroups) > 0 {
			sg.statements = append(sg.statements, fmt.Sprintf("INSERT INTO `%s` (%s) VALUES %s;", tableName, colList, strings.Join(valueGroups, ", ")))
		}
	}
}

func (sg *SqlGenerator) LogLoadDataLocalInfile(tableName string, columns []string, csvPath string) {
	if len(columns) == 0 || csvPath == "" {
		return
	}
	varCols := make([]string, len(columns))
	setClauses := make([]string, len(columns))
	for i, col := range columns {
		varName := fmt.Sprintf("@v%d", i)
		varCols[i] = varName
		setClauses[i] = fmt.Sprintf("`%s` = NULLIF(%s, '')", col, varName)
	}
	stmt := fmt.Sprintf("LOAD DATA LOCAL INFILE '%s' INTO TABLE `%s` FIELDS TERMINATED BY ',' ENCLOSED BY '"+`"`+"' LINES TERMINATED BY '\n' IGNORE 1 LINES ( %s ) SET %s;", csvPath, tableName, strings.Join(varCols, ", "), strings.Join(setClauses, ", "))
	sg.statements = append(sg.statements, stmt)
}

func (sg *SqlGenerator) LogCreateTempTable(tmpTable string, columns []string, idColumn string, df Frame) {
	if len(columns) == 0 {
		return
	}
	sg.statements = append(sg.statements, fmt.Sprintf("DROP TEMPORARY TABLE IF EXISTS `%s`;", tmpTable))
	cols := make([]string, 0, len(columns))
	for _, col := range columns {
		sqlType := sg.inferSQLType(df, col)
		if col == idColumn {
			sqlType = "BIGINT"
		}
		cols = append(cols, fmt.Sprintf("`%s` %s", col, sqlType))
	}
	create := fmt.Sprintf("CREATE TEMPORARY TABLE `%s` (%s);", tmpTable, strings.Join(cols, ", "))
	sg.statements = append(sg.statements, create)
}

func (sg *SqlGenerator) LogUpdateFromTempJoin(targetTable, tmpTable, idColumn string, columnNames []string) {
	if len(columnNames) == 0 {
		return
	}
	sets := make([]string, 0, len(columnNames))
	for _, col := range columnNames {
		sets = append(sets, fmt.Sprintf("t.`%s`=u.`%s`", col, col))
	}
	sql := fmt.Sprintf("UPDATE `%s` t JOIN `%s` u ON t.`%s`=u.`%s` SET %s;", targetTable, tmpTable, idColumn, idColumn, strings.Join(sets, ", "))
	sg.statements = append(sg.statements, sql)
}

func (sg *SqlGenerator) LogDropTempTable(tmpTable string) {
	sg.statements = append(sg.statements, fmt.Sprintf("DROP TEMPORARY TABLE IF EXISTS `%s`;", tmpTable))
}

// LogCreateTable optionally supports a partition clause passed via FrameColumnsPartitioner
// (an optional interface implemented by callers when special DDL is needed).
func (sg *SqlGenerator) LogCreateTable(tableName string, df Frame, columns []string) {
	if len(columns) == 0 {
		return
	}
	cols := []string{"`id` bigint NOT NULL AUTO_INCREMENT"}
	indexes := []string{}
	for _, colName := range columns {
		sqlType := sg.inferSQLType(df, colName)
		cols = append(cols, fmt.Sprintf("`%s` %s", colName, sqlType))
		indexes = append(indexes, fmt.Sprintf("KEY `idx_%s` (`%s`)", colName, colName))
	}
	partitionClause := ""
	if p, ok := df.(interface{ PartitionClause() string }); ok {
		partitionClause = strings.TrimSpace(p.PartitionClause())
	}
	createSQL := fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
        %s,
        %s,
        PRIMARY KEY (`+"`id`"+`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin %s`, tableName, strings.Join(cols, ", "), strings.Join(indexes, ", "), partitionClause)
	sg.statements = append(sg.statements, fmt.Sprintf("DROP TABLE IF EXISTS %s", tableName))
	sg.statements = append(sg.statements, createSQL)
}

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

func (sg *SqlGenerator) LogDropTable(tableName string) {
	sg.statements = append(sg.statements, fmt.Sprintf("DROP TABLE IF EXISTS `%s`", tableName))
}
func (sg *SqlGenerator) LogTruncateTable(tableName string) {
	sg.statements = append(sg.statements, fmt.Sprintf("TRUNCATE TABLE `%s`", tableName))
}
func (sg *SqlGenerator) LogAnalyzeTable(tableName string) {
	sg.statements = append(sg.statements, fmt.Sprintf("ANALYZE TABLE `%s` ALL COLUMNS", tableName))
}
func (sg *SqlGenerator) LogSetVariable(variable string, value interface{}) {
	sg.statements = append(sg.statements, fmt.Sprintf("SET %s = %s", variable, sg.formatValue(value)))
}
func (sg *SqlGenerator) LogUseDatabase(databaseName string) {
	sg.statements = append(sg.statements, fmt.Sprintf("USE `%s`", databaseName))
}
func (sg *SqlGenerator) LogComment(comment string) {
	for _, line := range strings.Split(comment, "\n") {
		sg.statements = append(sg.statements, fmt.Sprintf("-- %s", strings.TrimSpace(line)))
	}
}
func (sg *SqlGenerator) LogEmptyLine() { sg.statements = append(sg.statements, "") }
func (sg *SqlGenerator) Save(filename string) error {
	if len(sg.statements) == 0 {
		// Truncate file to avoid reusing stale SQL from previous runs
		return os.WriteFile(filename, []byte{}, 0644)
	}
	return os.WriteFile(filename, []byte(strings.Join(sg.statements, "\n")), 0644)
}
func (sg *SqlGenerator) GetStatements() []string { return sg.statements }
func (sg *SqlGenerator) Clear()                  { sg.statements = make([]string, 0) }
func (sg *SqlGenerator) Size() int               { return len(sg.statements) }

func (sg *SqlGenerator) formatValue(val interface{}) string {
	if val == nil {
		return "NULL"
	}
	switch v := val.(type) {
	case string:
		return fmt.Sprintf("'%s'", strings.ReplaceAll(v, "'", "''"))
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

func (sg *SqlGenerator) inferSQLType(df Frame, colName string) string {
	col := df.GetColumn(colName)
	if len(col) == 0 {
		return "VARCHAR(255)"
	}
	sampleSize := minInt(10, len(col))
	intCount, floatCount, dateCount, stringCount := 0, 0, 0, 0
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

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
