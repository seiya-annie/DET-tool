package data

import (
	"fmt"
	"math/rand"
	"strings"
	"time"
)

type DataGenerator struct{ rng *rand.Rand }

func NewDataGenerator() *DataGenerator {
	return &DataGenerator{rng: rand.New(rand.NewSource(time.Now().UnixNano()))}
}

func (dg *DataGenerator) appendValue(df *DataFrame, rowIndex int, value interface{}) {
	if rowIndex < len(df.data) {
		df.data[rowIndex] = append(df.data[rowIndex], value)
	} else {
		df.AddRow([]interface{}{value})
	}
}

func (dg *DataGenerator) Generate(modelConfig map[string]interface{}) *DataFrame {
	// modelConfig is expected to adhere to itypes.ModelConfig-like shape when used from main
	name := fmt.Sprintf("%v", modelConfig["Name"])
	modelType := fmt.Sprintf("%v", modelConfig["Type"])
	params, _ := modelConfig["Params"].(map[string]interface{})

	// Custom schema: partitioned skew table (user_id, user_name, created_time)
	if modelType == "partition_skew" {
		return dg.generatePartitionSkew(params)
	}

	tableName := name
	colInt := fmt.Sprintf("%s_int", tableName)
	colVarchar := fmt.Sprintf("%s_varchar", tableName)
	colDatetime := fmt.Sprintf("%s_datetime", tableName)

	rows := int(getFloatValue(params, "rows"))
	if rows == 0 {
		rows = 1000
	}
	df := NewDataFrame()

	if intRange, ok := params["int_range"].([]interface{}); ok && len(intRange) >= 2 {
		start := int(getFloatValue(map[string]interface{}{"val": intRange[0]}, "val"))
		end := int(getFloatValue(map[string]interface{}{"val": intRange[1]}, "val"))
		dg.generateIntColumn(df, colInt, modelType, start, end, rows, params)
	} else {
		df.AddColumn(colInt)
		for i := 0; i < rows; i++ {
			dg.appendValue(df, i, i+1)
		}
	}

	if varcharRange, ok := params["varchar_range"].(map[string]interface{}); ok {
		dg.generateVarcharColumn(df, colVarchar, varcharRange, rows)
	} else {
		df.AddColumn(colVarchar)
		for i := 0; i < rows; i++ {
			dg.appendValue(df, i, dg.generateRandomWord())
		}
	}

	if dateRange, ok := params["date_range"].([]interface{}); ok && len(dateRange) >= 2 {
		startStr := fmt.Sprintf("%v", dateRange[0])
		endStr := fmt.Sprintf("%v", dateRange[1])
		dg.generateDatetimeColumn(df, colDatetime, startStr, endStr, rows, params)
	} else {
		df.AddColumn(colDatetime)
		now := time.Now()
		for i := 0; i < rows; i++ {
			dg.appendValue(df, i, now)
		}
	}

	if modelType == "holes" {
		dg.applyHoles(df, params, colInt, colDatetime)
	}

	return df.Sample(1.0).ResetIndex(true)
}

// indexSlice returns [0,1,...,n-1]
func indexSlice(n int) []int {
	out := make([]int, n)
	for i := 0; i < n; i++ {
		out[i] = i
	}
	return out
}

// buildMonthlyPartitionClause builds PARTITION BY RANGE ( TO_DAYS(column) ) ... for given month starts
func BuildMonthlyPartitionClause(monthStarts []time.Time, col string) string {
	if len(monthStarts) == 0 {
		return ""
	}
	parts := make([]string, 0, len(monthStarts)+1)
	for i, m := range monthStarts {
		// partition name like p202401
		pname := fmt.Sprintf("p%04d%02d", m.Year(), m.Month())
		// upper bound is start of next month or maxvalue
		var upper string
		if i+1 < len(monthStarts) {
			upper = fmt.Sprintf("TO_DAYS('%s')", monthStarts[i+1].Format("2006-01-02"))
		} else {
			upper = "MAXVALUE"
		}
		parts = append(parts, fmt.Sprintf("PARTITION %s VALUES LESS THAN (%s)", pname, upper))
	}
	return "PARTITION BY RANGE (TO_DAYS(`" + col + "`)) (" + strings.Join(parts, ", ") + ")"
}

// generatePartitionSkew builds a 3-column dataset with monthly skewed distribution on created_time.
// Columns: partition_skew_id (int), partition_skew_varchar (UUID string), partition_skew_datetime (datetime)
func (dg *DataGenerator) generatePartitionSkew(params map[string]interface{}) *DataFrame {
	rows := int(getFloatValue(params, "rows"))
	if rows <= 0 {
		rows = 10000
	}
	// Parse date range; default to current year if missing
	startStr, endStr := "", ""
	if dr, ok := params["date_range"].([]interface{}); ok {
		if len(dr) >= 1 {
			startStr = fmt.Sprintf("%v", dr[0])
		}
		if len(dr) >= 2 {
			endStr = fmt.Sprintf("%v", dr[1])
		}
	}
	start, _ := time.Parse("2006-01-02", startStr)
	end, _ := time.Parse("2006-01-02", endStr)
	if start.IsZero() {
		now := time.Now()
		start = time.Date(now.Year(), 1, 1, 0, 0, 0, 0, time.UTC)
	}
	if end.IsZero() {
		end = start.AddDate(0, 6, 0)
	}
	// Align to month starts
	start = time.Date(start.Year(), start.Month(), 1, 0, 0, 0, 0, time.UTC)
	end = time.Date(end.Year(), end.Month(), 1, 0, 0, 0, 0, time.UTC)
	if !end.After(start) {
		end = start.AddDate(0, 1, 0)
	}

	// Build month buckets [start, nextStart)
	monthStarts := []time.Time{}
	for cur := start; cur.Before(end) || cur.Equal(end); cur = cur.AddDate(0, 1, 0) {
		monthStarts = append(monthStarts, cur)
		// stop after reaching end month
		if cur.Year() == end.Year() && cur.Month() == end.Month() {
			break
		}
	}
	if len(monthStarts) == 0 {
		monthStarts = append(monthStarts, start)
	}

	// Weights per month: accept optional month_weights or default descending; last month scaled down
	weights := []float64{}
	if mw, ok := params["month_weights"].([]interface{}); ok && len(mw) > 0 {
		for _, v := range mw {
			weights = append(weights, getFloatValue(map[string]interface{}{"val": v}, "val"))
		}
	}
	if len(weights) != len(monthStarts) {
		weights = make([]float64, len(monthStarts))
		for i := range weights {
			weights[i] = float64(len(monthStarts) - i)
		}
		if len(weights) > 0 {
			weights[len(weights)-1] = weights[len(weights)-1] * 0.25
			if weights[len(weights)-1] <= 0 {
				weights[len(weights)-1] = 0.1
			}
		}
	}
	probs := dg.buildProbabilityDistribution(weights, len(monthStarts))

	df := NewDataFrame()
	colID := "partition_skew_id"
	colVarchar := "partition_skew_varchar"
	colDatetime := "partition_skew_datetime"
	df.AddColumn(colID)
	df.AddColumn(colVarchar)
	df.AddColumn(colDatetime)

	for i := 0; i < rows; i++ {
		bucketIdx := dg.sampleFromDistribution(indexSlice(len(monthStarts)), probs)
		mStart := monthStarts[bucketIdx]
		nextStart := mStart.AddDate(0, 1, 0)
		// random day within the month bucket
		delta := nextStart.Sub(mStart)
		randDur := time.Duration(dg.rng.Int63n(int64(delta)))
		ts := mStart.Add(randDur)
		dg.appendValue(df, i, i+1)
		dg.appendValue(df, i, dg.generateUUIDString())
		dg.appendValue(df, i, ts)
	}

	part := BuildMonthlyPartitionClause(monthStarts, colDatetime)
	// Include partition key in PK to satisfy clustered index + partition constraints
	pks := []string{"id", colDatetime}
	return df.Sample(1.0).ResetIndex(true).WithPartitionClause(part).WithPrimaryKeys(pks)
}

func (dg *DataGenerator) generateIntColumn(df *DataFrame, colName string, modelType string, start, end, rows int, params map[string]interface{}) {
	df.AddColumn(colName)
	ndv := int(getFloatValue(params, "ndv"))
	if ndv == 0 || ndv > rows {
		ndv = rows
	}
	switch modelType {
	case "skew":
		dg.generateSkewedInt(df, start, end, rows, ndv, params)
	case "low_card":
		dg.generateLowCardinalityInt(df, start, end, rows, ndv)
	default:
		dg.generateUniformInt(df, start, end, rows, ndv)
	}
}

func (dg *DataGenerator) generateSkewedInt(df *DataFrame, start, end, rows, ndv int, params map[string]interface{}) {
	pool := make([]int, ndv)
	step := (end - start) / ndv
	if step == 0 {
		step = 1
	}
	for i := 0; i < ndv; i++ {
		val := start + i*step
		if val > end {
			val = end
		}
		pool[i] = val
	}
	for i := len(pool) - 1; i > 0; i-- {
		j := dg.rng.Intn(i + 1)
		pool[i], pool[j] = pool[j], pool[i]
	}
	weights := []float64{0.8, 0.2}
	if w, ok := params["skew_weights"].([]interface{}); ok {
		weights = make([]float64, len(w))
		for i, val := range w {
			weights[i] = getFloatValue(map[string]interface{}{"val": val}, "val")
		}
	}
	probDist := dg.buildProbabilityDistribution(weights, ndv)
	for i := 0; i < rows; i++ {
		value := dg.sampleFromDistribution(pool, probDist)
		dg.appendValue(df, i, value)
	}
}

func (dg *DataGenerator) generateLowCardinalityInt(df *DataFrame, start, end, rows, ndv int) {
	pool := make([]int, ndv)
	step := (end - start) / ndv
	if step == 0 {
		step = 1
	}
	for i := 0; i < ndv; i++ {
		val := start + i*step
		if val > end {
			val = end
		}
		pool[i] = val
	}
	for i := 0; i < rows; i++ {
		value := pool[dg.rng.Intn(len(pool))]
		dg.appendValue(df, i, value)
	}
}

func (dg *DataGenerator) generateUniformInt(df *DataFrame, start, end, rows, ndv int) {
	pool := make([]int, ndv)
	step := (end - start) / ndv
	if step == 0 {
		step = 1
	}
	for i := 0; i < ndv; i++ {
		val := start + i*step
		if val > end {
			val = end
		}
		pool[i] = val
	}
	for i := 0; i < rows; i++ {
		value := pool[dg.rng.Intn(len(pool))]
		dg.appendValue(df, i, value)
	}
}

func (dg *DataGenerator) generateVarcharColumn(df *DataFrame, colName string, varcharRange map[string]interface{}, rows int) {
	df.AddColumn(colName)
	if options, ok := varcharRange["options"].([]interface{}); ok {
		optionStrings := make([]string, len(options))
		for i, opt := range options {
			optionStrings[i] = fmt.Sprintf("%v", opt)
		}
		for i := 0; i < rows; i++ {
			value := optionStrings[dg.rng.Intn(len(optionStrings))]
			dg.appendValue(df, i, value)
		}
	} else {
		prefix := ""
		if p, ok := varcharRange["prefix"].(string); ok {
			prefix = p
		}
		suffixRange := []interface{}{1, rows}
		if sr, ok := varcharRange["suffix_range"].([]interface{}); ok && len(sr) >= 2 {
			suffixRange = sr
		}
		start := int(getFloatValue(map[string]interface{}{"val": suffixRange[0]}, "val"))
		end := int(getFloatValue(map[string]interface{}{"val": suffixRange[1]}, "val"))
		for i := 0; i < rows; i++ {
			suffix := dg.rng.Intn(end-start+1) + start
			value := fmt.Sprintf("%s%d", prefix, suffix)
			dg.appendValue(df, i, value)
		}
	}
}

func (dg *DataGenerator) generateDatetimeColumn(df *DataFrame, colName string, startStr, endStr string, rows int, params map[string]interface{}) {
	df.AddColumn(colName)
	start, err := time.Parse("2006-01-02", startStr)
	if err != nil {
		start = time.Now().AddDate(-1, 0, 0)
	}
	end, err := time.Parse("2006-01-02", endStr)
	if err != nil {
		end = time.Now()
	}
	ndv := int(getFloatValue(params, "ndv"))
	if ndv == 0 || ndv > rows {
		ndv = 100
	}
	datePool := dg.generateDatePool(start, end, ndv)
	for i := 0; i < rows; i++ {
		value := datePool[dg.rng.Intn(len(datePool))]
		dg.appendValue(df, i, value)
	}
}

func (dg *DataGenerator) applyHoles(df *DataFrame, params map[string]interface{}, colInt string, colDatetime string) {
	if intHoleRange, ok := params["int_hole_range"].([]interface{}); ok && len(intHoleRange) >= 2 {
		start := int(getFloatValue(map[string]interface{}{"val": intHoleRange[0]}, "val"))
		end := int(getFloatValue(map[string]interface{}{"val": intHoleRange[1]}, "val"))
		newData := [][]interface{}{}
		intColIndex := getColumnIndex(df, colInt)
		for _, row := range df.data {
			if intColIndex < len(row) {
				if intVal, ok := row[intColIndex].(int); ok {
					if intVal < start || intVal > end {
						newData = append(newData, row)
					}
				} else {
					newData = append(newData, row)
				}
			} else {
				newData = append(newData, row)
			}
		}
		df.data = newData
	}
	if dateHoleRange, ok := params["date_hole_range"].([]interface{}); ok && len(dateHoleRange) >= 2 {
		startStr := fmt.Sprintf("%v", dateHoleRange[0])
		endStr := fmt.Sprintf("%v", dateHoleRange[1])
		start, _ := time.Parse("2006-01-02", startStr)
		end, _ := time.Parse("2006-01-02", endStr)
		newData := [][]interface{}{}
		datetimeColIndex := getColumnIndex(df, colDatetime)
		for _, row := range df.data {
			if datetimeColIndex < len(row) {
				if timeVal, ok := row[datetimeColIndex].(time.Time); ok {
					if timeVal.Before(start) || timeVal.After(end) {
						newData = append(newData, row)
					}
				} else {
					newData = append(newData, row)
				}
			} else {
				newData = append(newData, row)
			}
		}
		df.data = newData
	}
}

// Helpers
func (dg *DataGenerator) buildProbabilityDistribution(weights []float64, size int) []float64 {
	sumW := 0.0
	for _, w := range weights {
		sumW += w
	}
	if sumW > 1.0 {
		for i := range weights {
			weights[i] = weights[i] / sumW
		}
	}
	remainProb := 1.0 - sumW
	remainCount := size - len(weights)
	if remainCount > 0 {
		remaining := make([]float64, remainCount)
		for i := range remaining {
			remaining[i] = remainProb / float64(remainCount)
		}
		return append(weights, remaining...)
	}
	result := weights[:size]
	sum := 0.0
	for _, w := range result {
		sum += w
	}
	for i := range result {
		result[i] = result[i] / sum
	}
	return result
}

func (dg *DataGenerator) generateDatePool(start, end time.Time, size int) []time.Time {
	if end.Before(start) {
		return []time.Time{start}
	}
	duration := end.Sub(start)
	days := int(duration.Hours() / 24)
	if days <= 0 {
		return []time.Time{start}
	}
	if size > days {
		size = days
	}
	result := make([]time.Time, size)
	step := float64(days) / float64(size-1)
	for i := 0; i < size; i++ {
		daysToAdd := int(float64(i) * step)
		result[i] = start.AddDate(0, 0, daysToAdd)
	}
	return result
}

func (dg *DataGenerator) sampleFromDistribution(values []int, probabilities []float64) int {
	r := dg.rng.Float64()
	cumulative := 0.0
	for i, prob := range probabilities {
		cumulative += prob
		if r <= cumulative {
			return values[i]
		}
	}
	return values[len(values)-1]
}

func (dg *DataGenerator) generateRandomWord() string {
	words := []string{"apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi", "lemon", "mango", "orange", "papaya", "quince", "raspberry"}
	return words[dg.rng.Intn(len(words))]
}

// generateUUIDString returns a short pseudo-UUID string suitable for SQL literals
func (dg *DataGenerator) generateUUIDString() string {
	b := make([]byte, 16)
	for i := range b {
		b[i] = byte(dg.rng.Intn(256))
	}
	// Format as 8-4-4-4-12 hex
	hex := func(x byte) string { return fmt.Sprintf("%02x", x) }
	parts := []string{
		hex(b[0]) + hex(b[1]) + hex(b[2]) + hex(b[3]),
		hex(b[4]) + hex(b[5]),
		hex(b[6]) + hex(b[7]),
		hex(b[8]) + hex(b[9]),
		hex(b[10]) + hex(b[11]) + hex(b[12]) + hex(b[13]) + hex(b[14]) + hex(b[15]),
	}
	return strings.Join(parts, "-")
}

func getFloatValue(m map[string]interface{}, key string) float64 {
	if m == nil {
		return 0
	}
	if val, ok := m[key]; ok {
		switch v := val.(type) {
		case float64:
			return v
		case int:
			return float64(v)
		case string:
			var f float64
			fmt.Sscanf(v, "%f", &f)
			return f
		}
	}
	return 0.0
}
