package main

import (
    datapkg "det-tool/internal/data"
)

// sqlLoggerAdapter adapts internal/data SQLLogger to our SqlGenerator adapter
type sqlLoggerAdapter struct{ gen *SqlGenerator }

func (a *sqlLoggerAdapter) LogComment(s string) { a.gen.LogComment(s) }

func (a *sqlLoggerAdapter) LogInsertBatch(table string, df *datapkg.DataFrame, batchSize int) { a.gen.LogInsertBatch(table, df, batchSize) }
func (a *sqlLoggerAdapter) LogLoadDataLocalInfile(table string, df *datapkg.DataFrame, csvPath string) { a.gen.LogLoadDataLocalInfile(table, df, csvPath) }
func (a *sqlLoggerAdapter) LogCreateTempTable(tmp string, df *datapkg.DataFrame, idCol string) { a.gen.LogCreateTempTable(tmp, df, idCol) }

func (a *sqlLoggerAdapter) LogUpdateFromTempJoin(table, tmp, idCol string, cols []string) {
    a.gen.LogUpdateFromTempJoin(table, tmp, idCol, cols)
}

func (a *sqlLoggerAdapter) LogDropTempTable(tmp string) { a.gen.LogDropTempTable(tmp) }

func (a *sqlLoggerAdapter) LogDeleteByIDs(table, idCol string, ids []interface{}, batch int) {
    a.gen.LogDeleteByIDs(table, idCol, ids, batch)
}

func (a *sqlLoggerAdapter) LogDeleteLimit(table string, limit int) { a.gen.LogDeleteLimit(table, limit) }
