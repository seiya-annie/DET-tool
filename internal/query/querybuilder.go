package query

import (
    "fmt"
    "os"
    "strings"
    "time"

    itypes "det-tool/internal/types"
)

type QueryBuilder struct{}
func NewQueryBuilder() *QueryBuilder { return &QueryBuilder{} }

func (qb *QueryBuilder) Generate(modelConfig itypes.ModelConfig, tableName string, outputFile string, currentStats map[string]map[string]interface{}) {
    oldContent := ""
    if _, err := os.Stat(outputFile); err == nil {
        content, err := os.ReadFile(outputFile)
        if err == nil && len(content) > 0 {
            timestamp := time.Now().Format("2006-01-02 15:04:05")
            oldContent = fmt.Sprintf("\n\n-- ========================================================\n-- [ARCHIVED HISTORY] Generated before %s\n-- ========================================================\n", timestamp)
            for _, line := range strings.Split(string(content), "\n") { oldContent += fmt.Sprintf("-- %s\n", line) }
        }
    }
    params := modelConfig.Params
    modelType := modelConfig.Type
    sqls := []string{fmt.Sprintf("-- Auto-generated for %s at %s", tableName, time.Now().Format("2006-01-02 15:04:05"))}
    colInt := fmt.Sprintf("%s_int", tableName)
    colStr := fmt.Sprintf("%s_varchar", tableName)
    colDt := fmt.Sprintf("%s_datetime", tableName)
    var minInt, maxInt int
    if currentStats != nil { if s, ok := currentStats[colInt]; ok { if v, ok := s["min"]; ok && v != nil { minInt = parseIntValue(v) }; if v, ok := s["max"]; ok && v != nil { maxInt = parseIntValue(v) } } }
    if minInt == 0 && maxInt == 0 {
        if intRange, ok := params["int_range"].([]interface{}); ok && len(intRange) >= 2 {
            minInt = int(getFloatValue(map[string]interface{}{"val": intRange[0]}, "val"))
            maxInt = int(getFloatValue(map[string]interface{}{"val": intRange[1]}, "val"))
        } else { minInt = 0; maxInt = 100 }
    }
    sqls = append(sqls, fmt.Sprintf("/* LABEL: int out of bound */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = %d", tableName, tableName, colInt, maxInt+1000))
    sqls = append(sqls, fmt.Sprintf("/* LABEL: int point lookup */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = %d", tableName, tableName, colInt, minInt+1))
    sqls = append(sqls, fmt.Sprintf("/* LABEL: int range scan */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s BETWEEN %d AND %d", tableName, tableName, colInt, minInt, minInt+50))
    if modelType == "holes" {
        if intHoleRange, ok := params["int_hole_range"].([]interface{}); ok && len(intHoleRange) >= 2 {
            holeStart := int(getFloatValue(map[string]interface{}{"val": intHoleRange[0]}, "val"))
            holeEnd := int(getFloatValue(map[string]interface{}{"val": intHoleRange[1]}, "val"))
            sqls = append(sqls, "-- [Int] Holes Specific Queries")
            sqls = append(sqls, fmt.Sprintf("/* LABEL: int in the hole */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s > %d AND %s < %d", tableName, tableName, colInt, holeStart, colInt, holeEnd))
            offset := (holeEnd - holeStart) / 10; if offset < 500 { offset = 500 }
            crossStart := minInt; if holeStart-offset > minInt { crossStart = holeStart - offset }
            crossEnd := holeStart + offset
            sqls = append(sqls, fmt.Sprintf("/* LABEL: int across hole */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s > %d AND %s < %d", tableName, tableName, colInt, crossStart, colInt, crossEnd))
        }
    }
    var prefix string
    var suffixStart, suffixEnd int
    if varcharRange, ok := params["varchar_range"].(map[string]interface{}); ok {
        if p, ok := varcharRange["prefix"].(string); ok { prefix = p }
        if suffixRange, ok := varcharRange["suffix_range"].([]interface{}); ok && len(suffixRange) >= 2 {
            suffixStart = int(getFloatValue(map[string]interface{}{"val": suffixRange[0]}, "val"))
            suffixEnd = int(getFloatValue(map[string]interface{}{"val": suffixRange[1]}, "val"))
        } else { suffixStart = 1; suffixEnd = 1000 }
    }
    sqls = append(sqls, fmt.Sprintf("/* LABEL: string out of bound */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = '%s%d'", tableName, tableName, colStr, prefix, suffixEnd+1000))
    sqls = append(sqls, fmt.Sprintf("/* LABEL: string point lookup */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = '%s%d'", tableName, tableName, colStr, prefix, suffixStart+1))
    sqls = append(sqls, fmt.Sprintf("/* LABEL: string range scan */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s BETWEEN '%s%d' AND '%s%d'", tableName, tableName, colStr, prefix, suffixStart, prefix, suffixStart+50))
    var dateMin, dateMax string
    if currentStats != nil { if dtStats, ok := currentStats[colDt]; ok { if maxVal, ok := dtStats["max"]; ok && maxVal != nil { dateMax = fmt.Sprintf("%v", maxVal) } } }
    if dateRange, ok := params["date_range"].([]interface{}); ok && len(dateRange) >= 2 {
        dateMin = fmt.Sprintf("%v", dateRange[0])
        if dateMax == "" { dateMax = fmt.Sprintf("%v", dateRange[1]) }
    } else { dateMin = "2024-01-01"; if dateMax == "" { dateMax = "2024-12-31" } }
    dtMin, err := time.Parse("2006-01-02", dateMin); if err != nil { dtMin = time.Now().AddDate(-1, 0, 0) }
    sqls = append(sqls, fmt.Sprintf("/* LABEL: datetime out of bound */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s > '%s'", tableName, tableName, colDt, dateMax))
    dtEq := dtMin.AddDate(0, 0, 1).Format("2006-01-02")
    sqls = append(sqls, fmt.Sprintf("/* LABEL: datetime point lookup */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = '%s'", tableName, tableName, colDt, dtEq))
    dtRangeEnd := dtMin.AddDate(0, 0, 30).Format("2006-01-02")
    sqls = append(sqls, fmt.Sprintf("/* LABEL: datetime range scan */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s BETWEEN '%s' AND '%s'", tableName, tableName, colDt, dateMin, dtRangeEnd))
    if modelType == "holes" {
        if dateHoleRange, ok := params["date_hole_range"].([]interface{}); ok && len(dateHoleRange) >= 2 {
            dhStartStr := fmt.Sprintf("%v", dateHoleRange[0])
            dhEndStr := fmt.Sprintf("%v", dateHoleRange[1])
            sqls = append(sqls, "-- [Datetime] Holes Specific Queries")
            sqls = append(sqls, fmt.Sprintf("/* LABEL: datetime in the hole */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s > '%s' AND %s < '%s'", tableName, tableName, colDt, dhStartStr, colDt, dhEndStr))
            dhStart, err1 := time.Parse("2006-01-02", dhStartStr); dhEnd, err2 := time.Parse("2006-01-02", dhEndStr)
            if err1 == nil && err2 == nil {
                gap := dhEnd.Sub(dhStart)
                offset := gap / 10; if offset < 24*time.Hour { offset = 24 * time.Hour }
                crossStart := dhStart.Add(-offset).Format("2006-01-02")
                crossEnd := dhStart.Add(offset).Format("2006-01-02")
                sqls = append(sqls, fmt.Sprintf("/* LABEL: datetime across hole */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s > '%s' AND %s < '%s'", tableName, tableName, colDt, crossStart, colDt, crossEnd))
            }
        }
    }
    sqls = append(sqls, fmt.Sprintf("/* LABEL: mixed condition */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s > %d AND %s LIKE '%s%%'", tableName, tableName, colInt, minInt, colStr, prefix))
    content := strings.Join(sqls, ";\n") + ";\n"
    if oldContent != "" { content += oldContent }
    if err := os.WriteFile(outputFile, []byte(content), 0644); err != nil { fmt.Printf("Error writing query file: %v\n", err) } else { fmt.Printf("Generated queries file: %s\n", outputFile) }
}

// helpers duplicated to avoid importing from main
func parseIntValue(val interface{}) int { switch v := val.(type) { case int: return v; case float64: return int(v); case string: var r int; fmt.Sscanf(v, "%d", &r); return r; default: return 0 } }
func getFloatValue(m map[string]interface{}, key string) float64 { if val, ok := m[key]; ok { switch v := val.(type) { case float64: return v; case int: return float64(v); case string: var f float64; fmt.Sscanf(v, "%f", &f); return f } }; return 0.0 }

