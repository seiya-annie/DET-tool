package main

import (
    datapkg "det-tool/internal/data"
    sqlgen "det-tool/internal/sqlgen"
)

// SqlGenerator is a thin adapter that preserves the original
// methods used across the main package, delegating to internal/sqlgen.
type SqlGenerator struct {
    inner *sqlgen.SqlGenerator
}

func NewSqlGenerator() *SqlGenerator { return &SqlGenerator{inner: sqlgen.NewSqlGenerator()} }

func (sg *SqlGenerator) LogDeleteLimit(tableName string, limitCount int) {
    sg.inner.LogDeleteLimit(tableName, limitCount)
}

func (sg *SqlGenerator) LogDeleteByIDs(tableName, idColumn string, ids []interface{}, batchSize int) {
    sg.inner.LogDeleteByIDs(tableName, idColumn, ids, batchSize)
}

func (sg *SqlGenerator) LogUpdate(tableName, idColumn string, ids []interface{}, columnNames []string, values [][]interface{}) {
    sg.inner.LogUpdate(tableName, idColumn, ids, columnNames, values)
}

func (sg *SqlGenerator) LogInsertBatch(tableName string, df *datapkg.DataFrame, batchSize int) {
    sg.inner.LogInsertBatch(tableName, df, df.Data(), df.Columns(), batchSize)
}

func (sg *SqlGenerator) LogLoadDataLocalInfile(tableName string, df *datapkg.DataFrame, csvPath string) {
    sg.inner.LogLoadDataLocalInfile(tableName, df.Columns(), csvPath)
}

func (sg *SqlGenerator) LogCreateTempTable(tmpTable string, df *datapkg.DataFrame, idColumn string) {
    sg.inner.LogCreateTempTable(tmpTable, df.Columns(), idColumn, df)
}

func (sg *SqlGenerator) LogUpdateFromTempJoin(targetTable, tmpTable, idColumn string, columnNames []string) {
    sg.inner.LogUpdateFromTempJoin(targetTable, tmpTable, idColumn, columnNames)
}

func (sg *SqlGenerator) LogDropTempTable(tmpTable string) { sg.inner.LogDropTempTable(tmpTable) }

func (sg *SqlGenerator) LogCreateTable(tableName string, df *datapkg.DataFrame) {
    sg.inner.LogCreateTable(tableName, df, df.Columns())
}

func (sg *SqlGenerator) LogCreateIndex(tableName string, columns []string, indexType string) {
    sg.inner.LogCreateIndex(tableName, columns, indexType)
}

func (sg *SqlGenerator) LogDropTable(tableName string) { sg.inner.LogDropTable(tableName) }
func (sg *SqlGenerator) LogTruncateTable(tableName string) { sg.inner.LogTruncateTable(tableName) }
func (sg *SqlGenerator) LogAnalyzeTable(tableName string) { sg.inner.LogAnalyzeTable(tableName) }

func (sg *SqlGenerator) LogSetVariable(variable string, value interface{}) { sg.inner.LogSetVariable(variable, value) }
func (sg *SqlGenerator) LogUseDatabase(databaseName string) { sg.inner.LogUseDatabase(databaseName) }
func (sg *SqlGenerator) LogComment(comment string) { sg.inner.LogComment(comment) }
func (sg *SqlGenerator) LogEmptyLine() { sg.inner.LogEmptyLine() }
func (sg *SqlGenerator) Save(filename string) error { return sg.inner.Save(filename) }
func (sg *SqlGenerator) GetStatements() []string { return sg.inner.GetStatements() }
func (sg *SqlGenerator) Clear() { sg.inner.Clear() }
func (sg *SqlGenerator) Size() int { return sg.inner.Size() }
