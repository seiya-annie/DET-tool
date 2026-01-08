package data

import (
    "fmt"
    "math/rand"
    "time"
)

type DataGenerator struct { rng *rand.Rand }

func NewDataGenerator() *DataGenerator { return &DataGenerator{rng: rand.New(rand.NewSource(time.Now().UnixNano()))} }

func (dg *DataGenerator) appendValue(df *DataFrame, rowIndex int, value interface{}) {
    if rowIndex < len(df.data) { df.data[rowIndex] = append(df.data[rowIndex], value) } else { df.AddRow([]interface{}{value}) }
}

func (dg *DataGenerator) Generate(modelConfig map[string]interface{}) *DataFrame {
    // modelConfig is expected to adhere to itypes.ModelConfig-like shape when used from main
    name := fmt.Sprintf("%v", modelConfig["Name"])
    modelType := fmt.Sprintf("%v", modelConfig["Type"])
    params, _ := modelConfig["Params"].(map[string]interface{})

    tableName := name
    colInt := fmt.Sprintf("%s_int", tableName)
    colVarchar := fmt.Sprintf("%s_varchar", tableName)
    colDatetime := fmt.Sprintf("%s_datetime", tableName)

    rows := int(getFloatValue(params, "rows"))
    if rows == 0 { rows = 1000 }
    df := NewDataFrame()

    if intRange, ok := params["int_range"].([]interface{}); ok && len(intRange) >= 2 {
        start := int(getFloatValue(map[string]interface{}{"val": intRange[0]}, "val"))
        end := int(getFloatValue(map[string]interface{}{"val": intRange[1]}, "val"))
        dg.generateIntColumn(df, colInt, modelType, start, end, rows, params)
    } else {
        df.AddColumn(colInt)
        for i := 0; i < rows; i++ { dg.appendValue(df, i, i+1) }
    }

    if varcharRange, ok := params["varchar_range"].(map[string]interface{}); ok { dg.generateVarcharColumn(df, colVarchar, varcharRange, rows) } else {
        df.AddColumn(colVarchar)
        for i := 0; i < rows; i++ { dg.appendValue(df, i, dg.generateRandomWord()) }
    }

    if dateRange, ok := params["date_range"].([]interface{}); ok && len(dateRange) >= 2 {
        startStr := fmt.Sprintf("%v", dateRange[0])
        endStr := fmt.Sprintf("%v", dateRange[1])
        dg.generateDatetimeColumn(df, colDatetime, startStr, endStr, rows, params)
    } else {
        df.AddColumn(colDatetime)
        now := time.Now()
        for i := 0; i < rows; i++ { dg.appendValue(df, i, now) }
    }

    if modelType == "holes" { dg.applyHoles(df, params, colInt, colDatetime) }

    result := df.Sample(1.0).ResetIndex(true)
    if datetimeCol := result.GetColumn(colDatetime); datetimeCol != nil {
        colIdx := getColumnIndex(result, colDatetime)
        if colIdx != -1 {
            for i, row := range result.data {
                if colIdx < len(row) {
                    if t, ok := row[colIdx].(time.Time); ok { result.data[i][colIdx] = t.Format("2006-01-02") }
                }
            }
        }
    }
    return result
}

func (dg *DataGenerator) generateIntColumn(df *DataFrame, colName string, modelType string, start, end, rows int, params map[string]interface{}) {
    df.AddColumn(colName)
    ndv := int(getFloatValue(params, "ndv"))
    if ndv == 0 || ndv > rows { ndv = rows }
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
    if step == 0 { step = 1 }
    for i := 0; i < ndv; i++ { val := start + i*step; if val > end { val = end }; pool[i] = val }
    for i := len(pool) - 1; i > 0; i-- { j := dg.rng.Intn(i + 1); pool[i], pool[j] = pool[j], pool[i] }
    weights := []float64{0.8, 0.2}
    if w, ok := params["skew_weights"].([]interface{}); ok { weights = make([]float64, len(w)); for i, val := range w { weights[i] = getFloatValue(map[string]interface{}{"val": val}, "val") } }
    probDist := dg.buildProbabilityDistribution(weights, ndv)
    for i := 0; i < rows; i++ { value := dg.sampleFromDistribution(pool, probDist); dg.appendValue(df, i, value) }
}

func (dg *DataGenerator) generateLowCardinalityInt(df *DataFrame, start, end, rows, ndv int) {
    pool := make([]int, ndv)
    step := (end - start) / ndv
    if step == 0 { step = 1 }
    for i := 0; i < ndv; i++ { val := start + i*step; if val > end { val = end }; pool[i] = val }
    for i := 0; i < rows; i++ { value := pool[dg.rng.Intn(len(pool))]; dg.appendValue(df, i, value) }
}

func (dg *DataGenerator) generateUniformInt(df *DataFrame, start, end, rows, ndv int) {
    pool := make([]int, ndv)
    step := (end - start) / ndv
    if step == 0 { step = 1 }
    for i := 0; i < ndv; i++ { val := start + i*step; if val > end { val = end }; pool[i] = val }
    for i := 0; i < rows; i++ { value := pool[dg.rng.Intn(len(pool))]; dg.appendValue(df, i, value) }
}

func (dg *DataGenerator) generateVarcharColumn(df *DataFrame, colName string, varcharRange map[string]interface{}, rows int) {
    df.AddColumn(colName)
    if options, ok := varcharRange["options"].([]interface{}); ok {
        optionStrings := make([]string, len(options))
        for i, opt := range options { optionStrings[i] = fmt.Sprintf("%v", opt) }
        for i := 0; i < rows; i++ { value := optionStrings[dg.rng.Intn(len(optionStrings))]; dg.appendValue(df, i, value) }
    } else {
        prefix := ""; if p, ok := varcharRange["prefix"].(string); ok { prefix = p }
        suffixRange := []interface{}{1, rows}
        if sr, ok := varcharRange["suffix_range"].([]interface{}); ok && len(sr) >= 2 { suffixRange = sr }
        start := int(getFloatValue(map[string]interface{}{"val": suffixRange[0]}, "val"))
        end := int(getFloatValue(map[string]interface{}{"val": suffixRange[1]}, "val"))
        for i := 0; i < rows; i++ { suffix := dg.rng.Intn(end-start+1) + start; value := fmt.Sprintf("%s%d", prefix, suffix); dg.appendValue(df, i, value) }
    }
}

func (dg *DataGenerator) generateDatetimeColumn(df *DataFrame, colName string, startStr, endStr string, rows int, params map[string]interface{}) {
    df.AddColumn(colName)
    start, err := time.Parse("2006-01-02", startStr); if err != nil { start = time.Now().AddDate(-1, 0, 0) }
    end, err := time.Parse("2006-01-02", endStr); if err != nil { end = time.Now() }
    ndv := int(getFloatValue(params, "ndv")); if ndv == 0 || ndv > rows { ndv = 100 }
    datePool := dg.generateDatePool(start, end, ndv)
    for i := 0; i < rows; i++ { value := datePool[dg.rng.Intn(len(datePool))]; dg.appendValue(df, i, value) }
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
                    if intVal < start || intVal > end { newData = append(newData, row) }
                } else { newData = append(newData, row) }
            } else { newData = append(newData, row) }
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
                    if timeVal.Before(start) || timeVal.After(end) { newData = append(newData, row) }
                } else { newData = append(newData, row) }
            } else { newData = append(newData, row) }
        }
        df.data = newData
    }
}

// Helpers
func (dg *DataGenerator) buildProbabilityDistribution(weights []float64, size int) []float64 {
    sumW := 0.0
    for _, w := range weights { sumW += w }
    if sumW > 1.0 { for i := range weights { weights[i] = weights[i] / sumW } }
    remainProb := 1.0 - sumW
    remainCount := size - len(weights)
    if remainCount > 0 {
        remaining := make([]float64, remainCount)
        for i := range remaining { remaining[i] = remainProb / float64(remainCount) }
        return append(weights, remaining...)
    }
    result := weights[:size]
    sum := 0.0
    for _, w := range result { sum += w }
    for i := range result { result[i] = result[i] / sum }
    return result
}

func (dg *DataGenerator) generateDatePool(start, end time.Time, size int) []time.Time {
    if end.Before(start) { return []time.Time{start} }
    duration := end.Sub(start)
    days := int(duration.Hours() / 24)
    if days <= 0 { return []time.Time{start} }
    if size > days { size = days }
    result := make([]time.Time, size)
    step := float64(days) / float64(size-1)
    for i := 0; i < size; i++ { daysToAdd := int(float64(i) * step); result[i] = start.AddDate(0, 0, daysToAdd) }
    return result
}

func (dg *DataGenerator) sampleFromDistribution(values []int, probabilities []float64) int {
    r := dg.rng.Float64(); cumulative := 0.0
    for i, prob := range probabilities { cumulative += prob; if r <= cumulative { return values[i] } }
    return values[len(values)-1]
}

func (dg *DataGenerator) generateRandomWord() string {
    words := []string{"apple","banana","cherry","date","elderberry","fig","grape","honeydew","kiwi","lemon","mango","orange","papaya","quince","raspberry"}
    return words[dg.rng.Intn(len(words))]
}

func getFloatValue(m map[string]interface{}, key string) float64 {
    if m == nil { return 0 }
    if val, ok := m[key]; ok {
        switch v := val.(type) {
        case float64:
            return v
        case int:
            return float64(v)
        case string:
            var f float64; fmt.Sscanf(v, "%f", &f); return f
        }
    }
    return 0.0
}


