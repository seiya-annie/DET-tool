package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// DataGenerator generates test data
type DataGenerator struct {
	rng *rand.Rand
}

// NewDataGenerator creates a new DataGenerator
func NewDataGenerator() *DataGenerator {
	return &DataGenerator{
		rng: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// appendValue appends a value to a specific row, creating the row if it doesn't exist
func (dg *DataGenerator) appendValue(df *DataFrame, rowIndex int, value interface{}) {
	if rowIndex < len(df.data) {
		// Row exists, append to it
		df.data[rowIndex] = append(df.data[rowIndex], value)
	} else {
		// Row doesn't exist, create new row
		df.AddRow([]interface{}{value})
	}
}

// Generate generates data based on model configuration
func (dg *DataGenerator) Generate(modelConfig ModelConfig) *DataFrame {
	modelType := modelConfig.Type
	params := modelConfig.Params

	rows := int(getFloatValue(params, "rows"))
	if rows == 0 {
		rows = 1000
	}

	df := NewDataFrame()

	// 1. Generate integer column
	if intRange, ok := params["int_range"].([]interface{}); ok && len(intRange) >= 2 {
		start := int(getFloatValue(map[string]interface{}{"val": intRange[0]}, "val"))
		end := int(getFloatValue(map[string]interface{}{"val": intRange[1]}, "val"))

		dg.generateIntColumn(df, modelType, start, end, rows, params)
	} else {
		// Default integer column
		df.AddColumn("col_int")
		for i := 0; i < rows; i++ {
			// Fix: Use appendValue instead of direct access which might panic or add new rows incorrectly
			dg.appendValue(df, i, i+1)
		}
	}

	// 2. Generate varchar column
	if varcharRange, ok := params["varchar_range"].(map[string]interface{}); ok {
		dg.generateVarcharColumn(df, varcharRange, rows)
	} else {
		// Default varchar column
		df.AddColumn("col_varchar")
		for i := 0; i < rows; i++ {
			// Fix: Use appendValue to ensure we attach to existing rows
			dg.appendValue(df, i, dg.generateRandomWord())
		}
	}

	// 3. Generate datetime column
	if dateRange, ok := params["date_range"].([]interface{}); ok && len(dateRange) >= 2 {
		startStr := fmt.Sprintf("%v", dateRange[0])
		endStr := fmt.Sprintf("%v", dateRange[1])
		dg.generateDatetimeColumn(df, startStr, endStr, rows, params)
	} else {
		// Default datetime column
		df.AddColumn("col_datetime")
		now := time.Now()
		for i := 0; i < rows; i++ {
			// Fix: Use appendValue to ensure we attach to existing rows
			dg.appendValue(df, i, now)
		}
	}

	// Apply holes if specified
	if modelType == "holes" {
		dg.applyHoles(df, params)
	}

	// Shuffle the data
	result := df.Sample(1.0).ResetIndex(true)

	// Convert datetime to string format for CSV compatibility
	if datetimeCol := result.GetColumn("col_datetime"); datetimeCol != nil {
		colIdx := getColumnIndex(result, "col_datetime")
		if colIdx != -1 {
			for i, row := range result.data {
				if colIdx < len(row) {
					if t, ok := row[colIdx].(time.Time); ok {
						result.data[i][colIdx] = t.Format("2006-01-02")
					}
				}
			}
		}
	}

	return result
}

// generateIntColumn generates integer column based on model type
func (dg *DataGenerator) generateIntColumn(df *DataFrame, modelType string, start, end, rows int, params map[string]interface{}) {
	df.AddColumn("col_int")

	// Calculate NDV (Number of Distinct Values)
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

// generateSkewedInt generates skewed integer data
func (dg *DataGenerator) generateSkewedInt(df *DataFrame, start, end, rows, ndv int, params map[string]interface{}) {
	// Create value pool
	pool := make([]int, ndv)
	step := (end - start) / ndv
	// Protect against div by zero if ndv is huge or range is small, though logic above handles ndv
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

	// Shuffle pool
	for i := len(pool) - 1; i > 0; i-- {
		j := dg.rng.Intn(i + 1)
		pool[i], pool[j] = pool[j], pool[i]
	}

	// Get skew weights
	weights := []float64{0.8, 0.2}
	if w, ok := params["skew_weights"].([]interface{}); ok {
		weights = make([]float64, len(w))
		for i, val := range w {
			weights[i] = getFloatValue(map[string]interface{}{"val": val}, "val")
		}
	}

	// Build probability distribution
	probDist := dg.buildProbabilityDistribution(weights, ndv)

	// Generate values
	for i := 0; i < rows; i++ {
		value := dg.sampleFromDistribution(pool, probDist)
		// Fix: Use appendValue
		dg.appendValue(df, i, value)
	}
}

// generateLowCardinalityInt generates low cardinality integer data
func (dg *DataGenerator) generateLowCardinalityInt(df *DataFrame, start, end, rows, ndv int) {
	// Create small pool of distinct values
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

	// Generate values from the small pool
	for i := 0; i < rows; i++ {
		value := pool[dg.rng.Intn(len(pool))] // Use len(pool) to be safe
		// Fix: Use appendValue
		dg.appendValue(df, i, value)
	}
}

// generateUniformInt generates uniformly distributed integer data
func (dg *DataGenerator) generateUniformInt(df *DataFrame, start, end, rows, ndv int) {
	// Create value pool
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

	// Generate values uniformly
	for i := 0; i < rows; i++ {
		value := pool[dg.rng.Intn(len(pool))]
		// Fix: Use appendValue
		dg.appendValue(df, i, value)
	}
}

// generateVarcharColumn generates varchar column
func (dg *DataGenerator) generateVarcharColumn(df *DataFrame, varcharRange map[string]interface{}, rows int) {
	df.AddColumn("col_varchar")

	if options, ok := varcharRange["options"].([]interface{}); ok {
		// Generate from predefined options
		optionStrings := make([]string, len(options))
		for i, opt := range options {
			optionStrings[i] = fmt.Sprintf("%v", opt)
		}

		for i := 0; i < rows; i++ {
			value := optionStrings[dg.rng.Intn(len(optionStrings))]
			// Fix: Use appendValue
			dg.appendValue(df, i, value)
		}
	} else {
		// Generate with prefix and suffix range
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
			// Fix: Use appendValue
			dg.appendValue(df, i, value)
		}
	}
}

// generateDatetimeColumn generates datetime column
func (dg *DataGenerator) generateDatetimeColumn(df *DataFrame, startStr, endStr string, rows int, params map[string]interface{}) {
	df.AddColumn("col_datetime")

	start, err := time.Parse("2006-01-02", startStr)
	if err != nil {
		start = time.Now().AddDate(-1, 0, 0)
	}

	end, err := time.Parse("2006-01-02", endStr)
	if err != nil {
		end = time.Now()
	}

	// Calculate NDV for datetime
	ndv := int(getFloatValue(params, "ndv"))
	if ndv == 0 || ndv > rows {
		ndv = 100 // Default NDV for datetime
	}

	// Generate date pool
	datePool := dg.generateDatePool(start, end, ndv)

	// Generate values
	for i := 0; i < rows; i++ {
		value := datePool[dg.rng.Intn(len(datePool))]
		// Fix: Use appendValue
		dg.appendValue(df, i, value)
	}
}

// applyHoles applies hole ranges to the data
func (dg *DataGenerator) applyHoles(df *DataFrame, params map[string]interface{}) {
	// Apply integer holes
	if intHoleRange, ok := params["int_hole_range"].([]interface{}); ok && len(intHoleRange) >= 2 {
		start := int(getFloatValue(map[string]interface{}{"val": intHoleRange[0]}, "val"))
		end := int(getFloatValue(map[string]interface{}{"val": intHoleRange[1]}, "val"))

		// Filter out rows in the hole range
		newData := [][]interface{}{}
		intColIndex := getColumnIndex(df, "col_int")

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

	// Apply datetime holes
	if dateHoleRange, ok := params["date_hole_range"].([]interface{}); ok && len(dateHoleRange) >= 2 {
		startStr := fmt.Sprintf("%v", dateHoleRange[0])
		endStr := fmt.Sprintf("%v", dateHoleRange[1])

		start, _ := time.Parse("2006-01-02", startStr)
		end, _ := time.Parse("2006-01-02", endStr)

		// Filter out rows in the hole range
		newData := [][]interface{}{}
		datetimeColIndex := getColumnIndex(df, "col_datetime")

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

// Helper functions

// buildProbabilityDistribution builds a probability distribution
func (dg *DataGenerator) buildProbabilityDistribution(weights []float64, size int) []float64 {
	sumW := 0.0
	for _, w := range weights {
		sumW += w
	}

	if sumW > 1.0 {
		// Normalize weights
		for i := range weights {
			weights[i] = weights[i] / sumW
		}
	}

	remainProb := 1.0 - sumW
	remainCount := size - len(weights)

	if remainCount > 0 {
		// Distribute remaining probability
		remaining := make([]float64, remainCount)
		for i := range remaining {
			remaining[i] = remainProb / float64(remainCount)
		}
		return append(weights, remaining...)
	} else {
		// Truncate and renormalize
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
}

// generateDatePool generates a pool of dates
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

// sampleFromDistribution samples from a weighted distribution
func (dg *DataGenerator) sampleFromDistribution(values []int, probabilities []float64) int {
	r := dg.rng.Float64()
	cumulative := 0.0

	for i, prob := range probabilities {
		cumulative += prob
		if r <= cumulative {
			return values[i]
		}
	}

	return values[len(values)-1] // Fallback to last value
}

// generateRandomWord generates a random word
func (dg *DataGenerator) generateRandomWord() string {
	words := []string{
		"apple", "banana", "cherry", "date", "elderberry",
		"fig", "grape", "honeydew", "kiwi", "lemon",
		"mango", "orange", "papaya", "quince", "raspberry",
	}
	return words[dg.rng.Intn(len(words))]
}

// Helper function to calculate linear interpolation
func lerp(start, end, t float64) float64 {
	return start + t*(end-start)
}

// Helper function to clamp a value between min and max
func clamp(value, min, max float64) float64 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}

// Helper function to calculate standard deviation
func stdDev(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	// Calculate mean
	var sum float64
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))

	// Calculate variance
	var variance float64
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	variance = variance / float64(len(values))

	return math.Sqrt(variance)
}
