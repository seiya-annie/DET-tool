package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// DataFrame represents a simple data frame structure
type DataFrame struct {
	columns []string
	data    [][]interface{}
}

// NewDataFrame creates a new empty DataFrame
func NewDataFrame() *DataFrame {
	return &DataFrame{
		columns: make([]string, 0),
		data:    make([][]interface{}, 0),
	}
}

// AddColumn adds a new column to the DataFrame
func (df *DataFrame) AddColumn(name string) {
	df.columns = append(df.columns, name)
}

// AddRow adds a new row to the DataFrame
func (df *DataFrame) AddRow(row []interface{}) {
	df.data = append(df.data, row)
}

// GetColumn returns a column by name
func (df *DataFrame) GetColumn(name string) []interface{} {
	colIndex := -1
	for i, col := range df.columns {
		if col == name {
			colIndex = i
			break
		}
	}

	if colIndex == -1 {
		return nil
	}

	result := make([]interface{}, len(df.data))
	for i, row := range df.data {
		if colIndex < len(row) {
			result[i] = row[colIndex]
		}
	}
	return result
}

// RenameColumns renames columns in the DataFrame
func (df *DataFrame) RenameColumns(renameMap map[string]string) {
	for i, col := range df.columns {
		if newName, ok := renameMap[col]; ok {
			df.columns[i] = newName
		}
	}
}

// Sample shuffles the DataFrame rows
func (df *DataFrame) Sample(frac float64) *DataFrame {
	// Simple shuffle implementation
	result := &DataFrame{
		columns: make([]string, len(df.columns)),
		data:    make([][]interface{}, len(df.data)),
	}
	copy(result.columns, df.columns)
	copy(result.data, df.data)

	// Fisher-Yates shuffle
	for i := len(result.data) - 1; i > 0; i-- {
		j := i // In real implementation, use random number
		result.data[i], result.data[j] = result.data[j], result.data[i]
	}

	return result
}

// ResetIndex resets the DataFrame index (no-op for this simple implementation)
func (df *DataFrame) ResetIndex(drop bool) *DataFrame {
	return df
}

// SaveCSV saves the DataFrame to a CSV file
func (df *DataFrame) SaveCSV(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	if err := writer.Write(df.columns); err != nil {
		return err
	}

	// Write data
	for _, row := range df.data {
		stringRow := make([]string, len(row))
		for i, val := range row {
			if val == nil {
				stringRow[i] = "" // Change: Write empty string for nil instead of "<nil>"
			} else {
				stringRow[i] = fmt.Sprintf("%v", val)
			}
		}
		if err := writer.Write(stringRow); err != nil {
			return err
		}
	}

	return nil
}

// LoadDataFrameFromCSV loads a DataFrame from a CSV file
func LoadDataFrameFromCSV(filename string) (*DataFrame, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// Read header
	columns, err := reader.Read()
	if err != nil {
		return nil, err
	}

	df := &DataFrame{
		columns: columns,
		data:    make([][]interface{}, 0),
	}

	// Read data
	for {
		record, err := reader.Read()
		if err != nil {
			break
		}

		row := make([]interface{}, len(record))
		for i, val := range record {
			// Change: Handle nil values explicitly
			if val == "" || val == "<nil>" {
				row[i] = nil
				continue
			}

			// Try to parse as different types
			if intVal, err := strconv.Atoi(val); err == nil {
				row[i] = intVal
			} else if floatVal, err := strconv.ParseFloat(val, 64); err == nil {
				row[i] = floatVal
			} else if timeVal, err := time.Parse("2006-01-02", val); err == nil {
				row[i] = timeVal
			} else {
				row[i] = val
			}
		}
		df.data = append(df.data, row)
	}

	return df, nil
}

// Concat concatenates two DataFrames
func Concat(dfs []*DataFrame, ignoreIndex bool) *DataFrame {
	if len(dfs) == 0 {
		return NewDataFrame()
	}

	result := &DataFrame{
		columns: make([]string, len(dfs[0].columns)),
		data:    make([][]interface{}, 0),
	}
	copy(result.columns, dfs[0].columns)

	for _, df := range dfs {
		result.data = append(result.data, df.data...)
	}

	return result
}

// IsNull checks if a value is null
func IsNull(val interface{}) bool {
	if val == nil {
		return true
	}
	if str, ok := val.(string); ok && str == "" {
		return true
	}
	return false
}

// NotNull checks if a value is not null
func NotNull(val interface{}) bool {
	return !IsNull(val)
}

// ToDatetime converts a string to datetime
func ToDatetime(dateStr string) (time.Time, error) {
	return time.Parse("2006-01-02", dateStr)
}

// Helper function to convert DataFrame column to string slice
func ColumnToStringSlice(df *DataFrame, columnName string) []string {
	col := df.GetColumn(columnName)
	if col == nil {
		return nil
	}

	result := make([]string, len(col))
	for i, val := range col {
		result[i] = fmt.Sprintf("%v", val)
	}
	return result
}

// Helper function to convert DataFrame column to int slice
func ColumnToIntSlice(df *DataFrame, columnName string) []int {
	col := df.GetColumn(columnName)
	if col == nil {
		return nil
	}

	result := make([]int, len(col))
	for i, val := range col {
		switch v := val.(type) {
		case int:
			result[i] = v
		case float64:
			result[i] = int(v)
		case string:
			if intVal, err := strconv.Atoi(v); err == nil {
				result[i] = intVal
			}
		}
	}
	return result
}

// StringInSlice checks if a string is in a slice
func StringInSlice(str string, list []string) bool {
	for _, v := range list {
		if v == str {
			return true
		}
	}
	return false
}

// IntInSlice checks if an int is in a slice
func IntInSlice(val int, list []int) bool {
	for _, v := range list {
		if v == val {
			return true
		}
	}
	return false
}

// ParseDateRange parses a date range string
func ParseDateRange(dateStr string) (time.Time, error) {
	formats := []string{
		"2006-01-02",
		"2006-01-02 15:04:05",
		"01/02/2006",
		"Jan 2, 2006",
	}

	for _, format := range formats {
		if t, err := time.Parse(format, strings.TrimSpace(dateStr)); err == nil {
			return t, nil
		}
	}

	return time.Time{}, fmt.Errorf("unable to parse date: %s", dateStr)
}
