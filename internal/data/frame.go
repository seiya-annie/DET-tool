package data

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

type DataFrame struct {
	columns         []string
	data            [][]interface{}
	partitionClause string
}

func NewDataFrame() *DataFrame {
	return &DataFrame{columns: make([]string, 0), data: make([][]interface{}, 0)}
}

func (df *DataFrame) AddColumn(name string) { df.columns = append(df.columns, name) }

func (df *DataFrame) AddRow(row []interface{}) { df.data = append(df.data, row) }

// Columns returns the column names (read-only copy)
func (df *DataFrame) Columns() []string {
	if df == nil {
		return nil
	}
	out := make([]string, len(df.columns))
	copy(out, df.columns)
	return out
}

// Data returns the underlying rows slice for read/write operations.
// Mutating the returned slice elements will affect the DataFrame.
func (df *DataFrame) Data() [][]interface{} {
	return df.data
}

// SetData replaces the internal rows with the provided slice.
func (df *DataFrame) SetData(rows [][]interface{}) {
	df.data = rows
}

// PartitionClause returns the partition clause to be appended to CREATE TABLE (if any)
func (df *DataFrame) PartitionClause() string { return df.partitionClause }

// WithPartitionClause attaches a raw partition clause to the frame (used by generators needing custom DDL)
func (df *DataFrame) WithPartitionClause(clause string) *DataFrame {
	df.partitionClause = clause
	return df
}

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

func (df *DataFrame) RenameColumns(renameMap map[string]string) {
	for i, col := range df.columns {
		if newName, ok := renameMap[col]; ok {
			df.columns[i] = newName
		}
	}
}

func (df *DataFrame) Sample(frac float64) *DataFrame {
	result := &DataFrame{columns: make([]string, len(df.columns)), data: make([][]interface{}, len(df.data))}
	copy(result.columns, df.columns)
	copy(result.data, df.data)
	for i := len(result.data) - 1; i > 0; i-- {
		j := i
		result.data[i], result.data[j] = result.data[j], result.data[i]
	}
	return result
}

func (df *DataFrame) ResetIndex(drop bool) *DataFrame { return df }

func (df *DataFrame) SaveCSV(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	defer writer.Flush()
	if err := writer.Write(df.columns); err != nil {
		return err
	}
	for _, row := range df.data {
		stringRow := make([]string, len(row))
		for i, val := range row {
			if val == nil {
				stringRow[i] = ""
			} else {
				switch v := val.(type) {
				case time.Time:
					stringRow[i] = v.Format("2006-01-02")
				default:
					stringRow[i] = fmt.Sprintf("%v", v)
				}
			}
		}
		if err := writer.Write(stringRow); err != nil {
			return err
		}
	}
	return nil
}

func LoadDataFrameFromCSV(filename string) (*DataFrame, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	reader := csv.NewReader(file)
	columns, err := reader.Read()
	if err != nil {
		return nil, err
	}
	df := &DataFrame{columns: columns, data: make([][]interface{}, 0)}
	for {
		record, err := reader.Read()
		if err != nil {
			break
		}
		row := make([]interface{}, len(record))
		for i, val := range record {
			if val == "" || val == "<nil>" {
				row[i] = nil
				continue
			}
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

func Concat(dfs []*DataFrame, ignoreIndex bool) *DataFrame {
	if len(dfs) == 0 {
		return NewDataFrame()
	}
	result := &DataFrame{columns: make([]string, len(dfs[0].columns)), data: make([][]interface{}, 0)}
	copy(result.columns, dfs[0].columns)
	for _, df := range dfs {
		result.data = append(result.data, df.data...)
	}
	return result
}

func IsNull(val interface{}) bool {
	if val == nil {
		return true
	}
	if s, ok := val.(string); ok && s == "" {
		return true
	}
	return false
}
func NotNull(val interface{}) bool                 { return !IsNull(val) }
func ToDatetime(dateStr string) (time.Time, error) { return time.Parse("2006-01-02", dateStr) }

func ColumnToStringSlice(df *DataFrame, columnName string) []string {
	col := df.GetColumn(columnName)
	if col == nil {
		return nil
	}
	result := make([]string, len(col))
	for i, v := range col {
		result[i] = fmt.Sprintf("%v", v)
	}
	return result
}
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
			if iv, err := strconv.Atoi(v); err == nil {
				result[i] = iv
			}
		}
	}
	return result
}

func StringInSlice(str string, list []string) bool {
	for _, v := range list {
		if v == str {
			return true
		}
	}
	return false
}
func IntInSlice(val int, list []int) bool {
	for _, v := range list {
		if v == val {
			return true
		}
	}
	return false
}

func ParseDateRange(dateStr string) (time.Time, error) {
	formats := []string{"2006-01-02", "2006-01-02 15:04:05", "01/02/2006", "Jan 2, 2006"}
	for _, f := range formats {
		if t, err := time.Parse(f, strings.TrimSpace(dateStr)); err == nil {
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("unable to parse date: %s", dateStr)
}

func getColumnIndex(df *DataFrame, columnName string) int {
	for i, col := range df.columns {
		if col == columnName {
			return i
		}
	}
	return -1
}

func (df *DataFrame) Size() int {
	if df == nil {
		return 0
	}
	return len(df.data)
}
