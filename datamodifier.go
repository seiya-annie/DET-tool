package main

import (
	"fmt"
	"math/rand"
)

// DataModifier modifies data based on incremental configuration
type DataModifier struct {
	sqlGen *SqlGenerator
	rng    *rand.Rand
}

// NewDataModifier creates a new DataModifier
func NewDataModifier(sqlGen *SqlGenerator) *DataModifier {
	return &DataModifier{
		sqlGen: sqlGen,
		rng:    rand.New(rand.NewSource(rand.Int63())),
	}
}

// Apply applies incremental changes to the data
func (dm *DataModifier) Apply(df *DataFrame, modelConfig ModelConfig, tableName string) *DataFrame {
	incremental := modelConfig.Incremental
	if incremental == nil {
		return df
	}

	// Clone the original DataFrame to avoid modifying it directly
	result := dm.cloneDataFrame(df)

	// Apply inserts
	insertRows := int(getFloatValue(incremental, "insert_rows"))
	if insertRows > 0 {
		fmt.Printf("    [Modifier] Inserting %d rows for %s\n", insertRows, tableName)
		result = dm.applyInserts(result, modelConfig, tableName, insertRows)
	}

	// Apply updates
	updateRatio := getFloatValue(incremental, "update_ratio")
	if updateRatio > 0 && len(result.data) > 0 {
		fmt.Printf("    [Modifier] Updating %.1f%% of rows for %s\n", updateRatio*100, tableName)
		result = dm.applyUpdates(result, modelConfig, tableName, updateRatio)
	}

	// Apply deletes
	deleteRatio := getFloatValue(incremental, "delete_ratio")
	if deleteRatio > 0 && len(result.data) > 0 {
		fmt.Printf("    [Modifier] Deleting %.1f%% of rows for %s\n", deleteRatio*100, tableName)
		result = dm.applyDeletes(result, tableName, deleteRatio)
	}

	return result
}

// cloneDataFrame creates a deep copy of the DataFrame
func (dm *DataModifier) cloneDataFrame(df *DataFrame) *DataFrame {
	cloned := &DataFrame{
		columns: make([]string, len(df.columns)),
		data:    make([][]interface{}, len(df.data)),
	}
	copy(cloned.columns, df.columns)

	for i, row := range df.data {
		cloned.data[i] = make([]interface{}, len(row))
		copy(cloned.data[i], row)
	}

	return cloned
}

// applyInserts applies insert operations
func (dm *DataModifier) applyInserts(df *DataFrame, modelConfig ModelConfig, tableName string, insertRows int) *DataFrame {
	// Create a temporary model config for generating new data
	tempConfig := ModelConfig{
		Name:        modelConfig.Name,
		Type:        modelConfig.Type,
		Description: fmt.Sprintf("Incremental data for %s", modelConfig.Name),
		Params:      make(map[string]interface{}),
		Incremental: make(map[string]interface{}),
	}

	// Copy parameters and override rows count
	for k, v := range modelConfig.Params {
		tempConfig.Params[k] = v
	}
	tempConfig.Params["rows"] = insertRows

	// Apply incremental parameters if specified
	for k, v := range modelConfig.Incremental {
		if k != "insert_rows" && k != "update_ratio" && k != "delete_ratio" {
			tempConfig.Params[k] = v
		}
	}

	// Generate new data
	// The generator will now automatically use tableName_int, tableName_varchar etc.
	dataGen := NewDataGenerator()
	newData := dataGen.Generate(tempConfig)

	// Log the INSERT statements
	if dm.sqlGen != nil {
		dm.sqlGen.LogComment(fmt.Sprintf("INSERT operations for %s", tableName))
		dm.sqlGen.LogInsertBatch(tableName, newData, 200)
	}

	// Concatenate the new data with existing data
	return Concat([]*DataFrame{df, newData}, true)
}

// applyUpdates applies update operations
func (dm *DataModifier) applyUpdates(df *DataFrame, modelConfig ModelConfig, tableName string, updateRatio float64) *DataFrame {
	if updateRatio <= 0 || updateRatio > 1 {
		return df
	}

	// Use correct column name prefix
	idColumn := fmt.Sprintf("%s_int", tableName)
	idColIndex := getColumnIndex(df, idColumn)

	if idColIndex == -1 {
		fmt.Printf("    [Warning] ID column %s not found for updates\n", idColumn)
		return df
	}

	// Find valid rows (with non-null IDs)
	validIndices := []int{}
	for i, row := range df.data {
		if idColIndex < len(row) && row[idColIndex] != nil {
			validIndices = append(validIndices, i)
		}
	}

	if len(validIndices) == 0 {
		return df
	}

	// Calculate number of rows to update
	updateCount := int(float64(len(validIndices)) * updateRatio)
	if updateCount == 0 {
		updateCount = 1 // Update at least one row
	}
	if updateCount > len(validIndices) {
		updateCount = len(validIndices)
	}

	// Randomly select rows to update
	selectedIndices := make([]int, updateCount)
	used := make(map[int]bool)

	for i := 0; i < updateCount; i++ {
		for {
			idx := validIndices[dm.rng.Intn(len(validIndices))]
			if !used[idx] {
				selectedIndices[i] = idx
				used[idx] = true
				break
			}
		}
	}

	// Create a temporary model config for generating update data
	tempConfig := ModelConfig{
		Name:        modelConfig.Name,
		Type:        modelConfig.Type,
		Description: fmt.Sprintf("Update data for %s", modelConfig.Name),
		Params:      make(map[string]interface{}),
		Incremental: make(map[string]interface{}),
	}

	// Copy parameters and override rows count
	for k, v := range modelConfig.Params {
		tempConfig.Params[k] = v
	}
	tempConfig.Params["rows"] = updateCount

	// Apply incremental parameters if specified
	for k, v := range modelConfig.Incremental {
		if k != "insert_rows" && k != "update_ratio" && k != "delete_ratio" {
			tempConfig.Params[k] = v
		}
	}

	// Generate new data for updates
	dataGen := NewDataGenerator()
	updateData := dataGen.Generate(tempConfig)

	// Collect IDs and new values for SQL generation
	ids := make([]interface{}, updateCount)
	columnNames := []string{}
	values := make([][]interface{}, updateCount)

	// Build column names (excluding ID column)
	for _, col := range df.columns {
		if col != idColumn {
			columnNames = append(columnNames, col)
		}
	}

	// Apply updates to the DataFrame and collect data for SQL
	for i, idx := range selectedIndices {
		if idx >= len(df.data) || i >= len(updateData.data) {
			continue
		}

		// Store ID for SQL
		ids[i] = df.data[idx][idColIndex]

		// Store new values for SQL
		values[i] = make([]interface{}, len(columnNames))

		// Apply updates to the DataFrame
		updateRow := updateData.data[i]
		for j, colName := range columnNames {
			colIndex := getColumnIndex(df, colName)
			if colIndex != -1 && colIndex < len(df.data[idx]) {
				if j < len(updateRow) {
					// Update DataFrame
					df.data[idx][colIndex] = updateRow[j]
					// Store for SQL
					values[i][j] = updateRow[j]
				} else {
					values[i][j] = df.data[idx][colIndex]
				}
			}
		}
	}

	// Log the UPDATE statements
	if dm.sqlGen != nil {
		dm.sqlGen.LogComment(fmt.Sprintf("UPDATE operations for %s", tableName))
		dm.sqlGen.LogUpdate(tableName, idColumn, ids, columnNames, values)
	}

	return df
}

// applyDeletes applies delete operations
func (dm *DataModifier) applyDeletes(df *DataFrame, tableName string, deleteRatio float64) *DataFrame {
	if deleteRatio <= 0 || deleteRatio > 1 {
		return df
	}

	currentTotal := len(df.data)
	if currentTotal == 0 {
		return df
	}

	deleteCount := int(float64(currentTotal) * deleteRatio)
	if deleteCount == 0 {
		deleteCount = 1 // Delete at least one row
	}
	if deleteCount > currentTotal {
		deleteCount = currentTotal
	}

	// Log the DELETE statements
	if dm.sqlGen != nil {
		dm.sqlGen.LogComment(fmt.Sprintf("DELETE operations for %s", tableName))
		dm.sqlGen.LogDeleteLimit(tableName, deleteCount)
	}

	// Randomly select rows to delete (mark as null/empty)
	if deleteCount < currentTotal {
		// Select random indices to delete
		deleteIndices := make(map[int]bool)
		for len(deleteIndices) < deleteCount {
			idx := dm.rng.Intn(currentTotal)
			deleteIndices[idx] = true
		}

		// Mark selected rows as deleted (using nil values)
		for idx := range deleteIndices {
			if idx < len(df.data) {
				// Replace row with nil values
				for j := range df.data[idx] {
					df.data[idx][j] = nil
				}
			}
		}
	} else {
		// Delete all rows
		for i := range df.data {
			for j := range df.data[i] {
				df.data[i][j] = nil
			}
		}
	}

	return df
}

// Helper functions
func isNull(val interface{}) bool {
	if val == nil {
		return true
	}
	if str, ok := val.(string); ok && str == "" {
		return true
	}
	return false
}

func randomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	result := make([]byte, length)
	for i := range result {
		result[i] = charset[rand.Intn(len(charset))]
	}
	return string(result)
}

func randomWord() string {
	words := []string{
		"apple", "banana", "cherry", "date", "elderberry",
		"fig", "grape", "honeydew", "kiwi", "lemon",
		"mango", "orange", "papaya", "quince", "raspberry",
	}
	return words[rand.Intn(len(words))]
}
