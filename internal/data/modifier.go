package data

import (
	"fmt"
	"math/rand"
	"path/filepath"
)

type DBRandIDs interface{ GetRandomIDs(string, int) []int64 }

type DataModifier struct {
	insertBatchSz int
	insertMode    string // insert|load
	rng           *rand.Rand
	tmpDir        string
	dbm           DBRandIDs
}

func NewDataModifier(insertBatchSize int, tmpDir string, dbm DBRandIDs) *DataModifier {
	if insertBatchSize <= 0 {
		insertBatchSize = 1000
	}
	return &DataModifier{insertBatchSz: insertBatchSize, insertMode: "insert", rng: rand.New(rand.NewSource(rand.Int63())), tmpDir: tmpDir, dbm: dbm}
}

// SetInsertMode sets insert mode to "insert" or "load".
func (dm *DataModifier) SetInsertMode(mode string) {
	if mode == "load" {
		dm.insertMode = "load"
	} else {
		dm.insertMode = "insert"
	}
}

// Adapter interface for logging SQL
type SQLLogger interface {
	LogComment(string)
	LogInsertBatch(table string, df *DataFrame, batchSize int)
	LogLoadDataLocalInfile(table string, df *DataFrame, csvPath string)
	LogCreateTempTable(tmp string, df *DataFrame, idCol string)
	LogUpdateFromTempJoin(table, tmp, idCol string, cols []string)
	LogDropTempTable(tmp string)
	LogDeleteByIDs(table, idCol string, ids []interface{}, batch int)
	LogDeleteLimit(table string, limit int)
}

// ApplyStats summarizes the actual changes applied during incremental modification
type ApplyStats struct {
	Inserted int
	Updated  int
	Deleted  int
	BaseRows int
}

func (dm *DataModifier) Apply(df *DataFrame, modelConfig map[string]interface{}, tableName string, logger SQLLogger) (*DataFrame, ApplyStats) {
	inc, _ := modelConfig["Incremental"].(map[string]interface{})
	if inc == nil {
		return df, ApplyStats{BaseRows: len(df.data)}
	}
	stats := ApplyStats{BaseRows: len(df.data)}
	result := dm.cloneDataFrame(df)
	insertRows := int(getFloatValue(inc, "insert_rows"))
	if insertRows > 0 {
		result = dm.applyInserts(result, modelConfig, tableName, insertRows, logger)
		stats.Inserted = insertRows
	}
	// Use BaseRows as target basis for update/delete counts; convert to ratio on current size
	baseRows := stats.BaseRows
	updateRatio := getFloatValue(inc, "update_ratio")
	if updateRatio > 0 && len(result.data) > 0 {
		targetUpdate := int(float64(baseRows) * updateRatio)
		if targetUpdate == 0 && updateRatio > 0 {
			targetUpdate = 1
		}
		ratioOnCurrent := float64(targetUpdate) / float64(len(result.data))
		if ratioOnCurrent > 1 {
			ratioOnCurrent = 1
		}
		var up int
		result, up = dm.applyUpdates(result, modelConfig, tableName, ratioOnCurrent, logger)
		stats.Updated = up
	}
	deleteRatio := getFloatValue(inc, "delete_ratio")
	if deleteRatio > 0 && len(result.data) > 0 {
		targetDelete := int(float64(baseRows) * deleteRatio)
		if targetDelete == 0 && deleteRatio > 0 {
			targetDelete = 1
		}
		ratioOnCurrent := float64(targetDelete) / float64(len(result.data))
		if ratioOnCurrent > 1 {
			ratioOnCurrent = 1
		}
		var del int
		result, del = dm.applyDeletes(result, tableName, ratioOnCurrent, logger)
		stats.Deleted = del
	}
	return result, stats
}

func (dm *DataModifier) cloneDataFrame(df *DataFrame) *DataFrame {
	cloned := &DataFrame{columns: make([]string, len(df.columns)), data: make([][]interface{}, len(df.data))}
	copy(cloned.columns, df.columns)
	for i, row := range df.data {
		cloned.data[i] = make([]interface{}, len(row))
		copy(cloned.data[i], row)
	}
	return cloned
}

func (dm *DataModifier) applyInserts(df *DataFrame, modelConfig map[string]interface{}, tableName string, insertRows int, logger SQLLogger) *DataFrame {
	// Convert to simple map config for generator
	cfg := map[string]interface{}{"Name": modelConfig["Name"], "Type": modelConfig["Type"], "Params": map[string]interface{}{}, "Incremental": map[string]interface{}{}}
	if p, ok := modelConfig["Params"].(map[string]interface{}); ok {
		for k, v := range p {
			cfg["Params"].(map[string]interface{})[k] = v
		}
	}
	cfg["Params"].(map[string]interface{})["rows"] = insertRows
	if inc, ok := modelConfig["Incremental"].(map[string]interface{}); ok {
		for k, v := range inc {
			if k != "insert_rows" && k != "update_ratio" && k != "delete_ratio" {
				cfg["Params"].(map[string]interface{})[k] = v
			}
		}
	}
	gen := NewDataGenerator()
	newData := gen.Generate(cfg)
	if logger != nil {
		logger.LogComment(fmt.Sprintf("INSERT operations for %s", tableName))
		if dm.insertMode == "load" {
			csvName := filepath.Join(dm.tmpDir, fmt.Sprintf("inc_insert_%s.csv", tableName))
			abs, _ := filepath.Abs(csvName)
			if err := newData.SaveCSV(csvName); err == nil {
				logger.LogLoadDataLocalInfile(tableName, newData, abs)
			} else {
				logger.LogInsertBatch(tableName, newData, dm.insertBatchSz)
			}
		} else {
			logger.LogInsertBatch(tableName, newData, dm.insertBatchSz)
		}
	}
	return Concat([]*DataFrame{df, newData}, true)
}

func (dm *DataModifier) applyUpdates(df *DataFrame, modelConfig map[string]interface{}, tableName string, updateRatio float64, logger SQLLogger) (*DataFrame, int) {
	if updateRatio <= 0 || updateRatio > 1 {
		return df, 0
	}
	// Prefer <table>_int; fallback to PK column "id"; otherwise update all rows.
	idColumn := fmt.Sprintf("%s_int", tableName)
	idColIndex := getColumnIndex(df, idColumn)
	if idColIndex < 0 {
		if alt := getColumnIndex(df, "id"); alt >= 0 {
			idColumn = "id"
			idColIndex = alt
		}
	}
	valid := []int{}
	if idColIndex >= 0 {
		for i, row := range df.data {
			if idColIndex < len(row) && row[idColIndex] != nil {
				valid = append(valid, i)
			}
		}
	} else {
		for i := range df.data {
			valid = append(valid, i)
		}
	}
	if len(valid) == 0 {
		return df, 0
	}
	updateCount := int(float64(len(valid)) * updateRatio)
	if updateCount == 0 {
		updateCount = 1
	}
	if updateCount > len(valid) {
		updateCount = len(valid)
	}
	selected := make([]int, updateCount)
	used := make(map[int]bool)
	for i := 0; i < updateCount; i++ {
		for {
			idx := valid[dm.rng.Intn(len(valid))]
			if !used[idx] {
				selected[i] = idx
				used[idx] = true
				break
			}
		}
	}
	cfg := map[string]interface{}{"Name": modelConfig["Name"], "Type": modelConfig["Type"], "Params": map[string]interface{}{}, "Incremental": map[string]interface{}{}}
	if p, ok := modelConfig["Params"].(map[string]interface{}); ok {
		for k, v := range p {
			cfg["Params"].(map[string]interface{})[k] = v
		}
	}
	cfg["Params"].(map[string]interface{})["rows"] = updateCount
	if inc, ok := modelConfig["Incremental"].(map[string]interface{}); ok {
		for k, v := range inc {
			if k != "insert_rows" && k != "update_ratio" && k != "delete_ratio" {
				cfg["Params"].(map[string]interface{})[k] = v
			}
		}
	}
	gen := NewDataGenerator()
	updateData := gen.Generate(cfg)
	columnNames := make([]string, 0, len(df.columns))
	if idColIndex >= 0 {
		for _, c := range df.columns {
			if c != idColumn {
				columnNames = append(columnNames, c)
			}
		}
	} else {
		for _, c := range df.columns {
			columnNames = append(columnNames, c)
		}
	}
	var ids []int64
	if dm.dbm != nil {
		ids = dm.dbm.GetRandomIDs(tableName, updateCount)
	}
	tmp := NewDataFrame()
	tmp.AddColumn("id")
	for _, c := range df.columns {
		tmp.AddColumn(c)
	}
	for i := 0; i < updateCount && i < len(ids) && i < len(selected) && i < len(updateData.data); i++ {
		idx := selected[i]
		updateRow := updateData.data[i]
		for _, colName := range columnNames {
			colIndex := getColumnIndex(df, colName)
			srcIndex := getColumnIndex(updateData, colName)
			if colIndex != -1 && srcIndex != -1 && colIndex < len(df.data[idx]) && srcIndex < len(updateRow) {
				df.data[idx][colIndex] = updateRow[srcIndex]
			}
		}
		tmpRow := make([]interface{}, 1+len(df.columns))
		tmpRow[0] = ids[i]
		copy(tmpRow[1:], df.data[idx])
		tmp.AddRow(tmpRow)
	}
	if logger != nil && tmp.Size() > 0 {
		logger.LogComment(fmt.Sprintf("UPDATE operations for %s (temp table join)", tableName))
		tmpName := fmt.Sprintf("tmp_%s_update", tableName)
		logger.LogCreateTempTable(tmpName, tmp, "id")
		csvName := filepath.Join(dm.tmpDir, fmt.Sprintf("inc_update_%s.csv", tableName))
		abs, _ := filepath.Abs(csvName)
		if err := tmp.SaveCSV(csvName); err == nil {
			logger.LogLoadDataLocalInfile(tmpName, tmp, abs)
			logger.LogUpdateFromTempJoin(tableName, tmpName, "id", columnNames)
			logger.LogDropTempTable(tmpName)
		}
	}
	return df, updateCount
}

func (dm *DataModifier) applyDeletes(df *DataFrame, tableName string, deleteRatio float64, logger SQLLogger) (*DataFrame, int) {
	if deleteRatio <= 0 || deleteRatio > 1 {
		return df, 0
	}
	currentTotal := len(df.data)
	if currentTotal == 0 {
		return df, 0
	}
	deleteCount := int(float64(currentTotal) * deleteRatio)
	if deleteCount == 0 {
		deleteCount = 1
	}
	if deleteCount > currentTotal {
		deleteCount = currentTotal
	}
	if logger != nil {
		logger.LogComment(fmt.Sprintf("DELETE operations for %s", tableName))
		if dm.dbm != nil {
			ids := dm.dbm.GetRandomIDs(tableName, deleteCount)
			if len(ids) > 0 {
				idsIfc := make([]interface{}, len(ids))
				for i, v := range ids {
					idsIfc[i] = v
				}
				logger.LogDeleteByIDs(tableName, "id", idsIfc, 1000)
			} else {
				logger.LogDeleteLimit(tableName, deleteCount)
			}
		} else {
			logger.LogDeleteLimit(tableName, deleteCount)
		}
	}
	if deleteCount < currentTotal {
		deleteIndices := make(map[int]bool)
		for len(deleteIndices) < deleteCount {
			idx := dm.rng.Intn(currentTotal)
			deleteIndices[idx] = true
		}
		for idx := range deleteIndices {
			if idx < len(df.data) {
				for j := range df.data[idx] {
					df.data[idx][j] = nil
				}
			}
		}
	} else {
		for i := range df.data {
			for j := range df.data[i] {
				df.data[i][j] = nil
			}
		}
	}
	return df, deleteCount
}
