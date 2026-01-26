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
			for _, line := range strings.Split(string(content), "\n") {
				oldContent += fmt.Sprintf("-- %s\n", line)
			}
		}
	}
	params := modelConfig.Params
	modelType := modelConfig.Type
	sqls := []string{fmt.Sprintf("-- Auto-generated for %s at %s", tableName, time.Now().Format("2006-01-02 15:04:05"))}
	colInt := fmt.Sprintf("%s_int", tableName)
	colStr := fmt.Sprintf("%s_varchar", tableName)
	colDt := fmt.Sprintf("%s_datetime", tableName)
	if strings.EqualFold(modelType, "partition_skew") {
		colInt = "partition_skew_id"
		colStr = "partition_skew_varchar"
		colDt = "partition_skew_datetime"
	}
	var minInt, maxInt int
	if currentStats != nil {
		if s, ok := currentStats[colInt]; ok {
			if v, ok := s["min"]; ok && v != nil {
				minInt = parseIntValue(v)
			}
			if v, ok := s["max"]; ok && v != nil {
				maxInt = parseIntValue(v)
			}
		}
	}
	if minInt == 0 && maxInt == 0 {
		if intRange, ok := params["int_range"].([]interface{}); ok && len(intRange) >= 2 {
			minInt = int(getFloatValue(map[string]interface{}{"val": intRange[0]}, "val"))
			maxInt = int(getFloatValue(map[string]interface{}{"val": intRange[1]}, "val"))
		} else {
			minInt = 0
			maxInt = 100
		}
	}
	// Base (initial) min/max strictly from config for histogram-boundary queries
	baseMinInt, baseMaxInt := minInt, maxInt
	if intRange, ok := params["int_range"].([]interface{}); ok && len(intRange) >= 2 {
		baseMinInt = int(getFloatValue(map[string]interface{}{"val": intRange[0]}, "val"))
		baseMaxInt = int(getFloatValue(map[string]interface{}{"val": intRange[1]}, "val"))
	}
	sqls = append(sqls, fmt.Sprintf("/* LABEL: int out of bound */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = %d", tableName, tableName, colInt, maxInt+1000))
	sqls = append(sqls, fmt.Sprintf("/* LABEL: int point lookup */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = %d", tableName, tableName, colInt, minInt+1))
	sqls = append(sqls, fmt.Sprintf("/* LABEL: int range scan */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s BETWEEN %d AND %d", tableName, tableName, colInt, minInt, minInt+50))
	// Use base min/max for histogram-boundary queries
	sqls = append(sqls, fmt.Sprintf("/* LABEL: int last value in last histogram */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = %d", tableName, tableName, colInt, baseMaxInt))
	sqls = append(sqls, fmt.Sprintf("/* LABEL: int first value in first histogram */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = %d", tableName, tableName, colInt, baseMinInt))

	if modelType == "holes" {
		if intHoleRange, ok := params["int_hole_range"].([]interface{}); ok && len(intHoleRange) >= 2 {
			holeStart := int(getFloatValue(map[string]interface{}{"val": intHoleRange[0]}, "val"))
			holeEnd := int(getFloatValue(map[string]interface{}{"val": intHoleRange[1]}, "val"))
			sqls = append(sqls, "-- [Int] Holes Specific Queries")
			sqls = append(sqls, fmt.Sprintf("/* LABEL: int in the hole */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s > %d AND %s < %d", tableName, tableName, colInt, holeStart, colInt, holeEnd))
			offset := (holeEnd - holeStart) / 10
			if offset < 500 {
				offset = 500
			}
			crossStart := minInt
			if holeStart-offset > minInt {
				crossStart = holeStart - offset
			}
			crossEnd := holeStart + offset
			sqls = append(sqls, fmt.Sprintf("/* LABEL: int across hole */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s > %d AND %s < %d", tableName, tableName, colInt, crossStart, colInt, crossEnd))
			// New: int include hole (outside the hole interval with small margins)
			left := holeStart - 10
			right := holeEnd + 10
			sqls = append(sqls, fmt.Sprintf("/* LABEL: int include hole */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s < %d OR %s > %d", tableName, tableName, colInt, left, colInt, right))
		}
	}
	var prefix string
	var suffixStart, suffixEnd int
	if varcharRange, ok := params["varchar_range"].(map[string]interface{}); ok {
		if p, ok := varcharRange["prefix"].(string); ok {
			prefix = p
		}
		if suffixRange, ok := varcharRange["suffix_range"].([]interface{}); ok && len(suffixRange) >= 2 {
			suffixStart = int(getFloatValue(map[string]interface{}{"val": suffixRange[0]}, "val"))
			suffixEnd = int(getFloatValue(map[string]interface{}{"val": suffixRange[1]}, "val"))
		} else {
			suffixStart = 1
			suffixEnd = 1000
		}
	}
	sqls = append(sqls, fmt.Sprintf("/* LABEL: string out of bound */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = '%s%d'", tableName, tableName, colStr, prefix, suffixEnd+1000))
	sqls = append(sqls, fmt.Sprintf("/* LABEL: string point lookup */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = '%s%d'", tableName, tableName, colStr, prefix, suffixStart+1))
	sqls = append(sqls, fmt.Sprintf("/* LABEL: string range scan */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s BETWEEN '%s%d' AND '%s%d'", tableName, tableName, colStr, prefix, suffixStart, prefix, suffixStart+50))
	sqls = append(sqls, fmt.Sprintf("/* LABEL: string last value in last histogram */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = '%s%d'", tableName, tableName, colStr, prefix, suffixEnd))
	sqls = append(sqls, fmt.Sprintf("/* LABEL: string first value in first histogram */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = '%s%d'", tableName, tableName, colStr, prefix, suffixStart))
	var dateMin, dateMax string
	if currentStats != nil {
		if dtStats, ok := currentStats[colDt]; ok {
			if maxVal, ok := dtStats["max"]; ok && maxVal != nil {
				dateMax = fmt.Sprintf("%v", maxVal)
			}
		}
	}
	// Base (initial) date range strictly from config for histogram-boundary queries
	baseDateMin := "2024-01-01"
	baseDateMax := "2024-12-31"
	if dateRange, ok := params["date_range"].([]interface{}); ok && len(dateRange) >= 2 {
		baseDateMin = fmt.Sprintf("%v", dateRange[0])
		baseDateMax = fmt.Sprintf("%v", dateRange[1])
	}
	// For adaptive queries, use baseDateMin for start; prefer currentStats max as end if present, else baseDateMax
	dateMin = baseDateMin
	if dateMax == "" {
		dateMax = baseDateMax
	}
	// Guard: ensure dateMax is a valid date; otherwise fall back to baseDateMax
	if _, err := time.Parse("2006-01-02", strings.TrimSpace(dateMax)); err != nil {
		dateMax = baseDateMax
	}
	dtMin, err := time.Parse("2006-01-02", dateMin)
	if err != nil {
		dtMin = time.Now().AddDate(-1, 0, 0)
	}
	sqls = append(sqls, fmt.Sprintf("/* LABEL: datetime out of bound */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s > '%s'", tableName, tableName, colDt, dateMax))
	dtEq := dtMin.AddDate(0, 0, 1).Format("2006-01-02")
	sqls = append(sqls, fmt.Sprintf("/* LABEL: datetime point lookup */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = '%s'", tableName, tableName, colDt, dtEq))
	dtRangeEnd := dtMin.AddDate(0, 0, 30).Format("2006-01-02")
	sqls = append(sqls, fmt.Sprintf("/* LABEL: datetime range scan */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s BETWEEN '%s' AND '%s'", tableName, tableName, colDt, dateMin, dtRangeEnd))
	// Cross-year query based on base date range year
	acrossYearStart := time.Date(dtMin.Year(), time.December, 31, 0, 0, 0, 0, time.UTC).Format("2006-01-02")
	acrossYearEnd := time.Date(dtMin.Year()+1, time.January, 1, 0, 0, 0, 0, time.UTC).Format("2006-01-02")
	sqls = append(sqls, fmt.Sprintf("/* LABEL: datetime across year */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s BETWEEN '%s' AND '%s'", tableName, tableName, colDt, acrossYearStart, acrossYearEnd))
	// Use base init min/max for histogram-boundary queries
	sqls = append(sqls, fmt.Sprintf("/* LABEL: datetime last value in last histogram */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = '%s'", tableName, tableName, colDt, baseDateMax))
	sqls = append(sqls, fmt.Sprintf("/* LABEL: datetime first value in first histogram */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = '%s'", tableName, tableName, colDt, baseDateMin))
	if modelType == "holes" {
		if dateHoleRange, ok := params["date_hole_range"].([]interface{}); ok && len(dateHoleRange) >= 2 {
			dhStartStr := fmt.Sprintf("%v", dateHoleRange[0])
			dhEndStr := fmt.Sprintf("%v", dateHoleRange[1])
			dhStartStr = strings.TrimSpace(dhStartStr)
			dhEndStr = strings.TrimSpace(dhEndStr)
			sqls = append(sqls, "-- [Datetime] Holes Specific Queries")
			sqls = append(sqls, fmt.Sprintf("/* LABEL: datetime in the hole */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s > '%s' AND %s < '%s'", tableName, tableName, colDt, dhStartStr, colDt, dhEndStr))
			dhStart, err1 := time.Parse("2006-01-02", dhStartStr)
			dhEnd, err2 := time.Parse("2006-01-02", dhEndStr)
			if err1 == nil && err2 == nil {
				gap := dhEnd.Sub(dhStart)
				offset := gap / 10
				if offset < 24*time.Hour {
					offset = 24 * time.Hour
				}
				crossStart := dhStart.Add(-offset).Format("2006-01-02")
				crossEnd := dhStart.Add(offset).Format("2006-01-02")
				sqls = append(sqls, fmt.Sprintf("/* LABEL: datetime across hole */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s > '%s' AND %s < '%s'", tableName, tableName, colDt, crossStart, colDt, crossEnd))
				left := dhStart.AddDate(0, 0, -1).Format("2006-01-02")
				right := dhEnd.AddDate(0, 0, 1).Format("2006-01-02")
				sqls = append(sqls, fmt.Sprintf("/* LABEL: datetime include hole */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s < '%s' OR %s > '%s'", tableName, tableName, colDt, left, colDt, right))
			}
		}
	}
	sqls = append(sqls, fmt.Sprintf("/* LABEL: mixed condition */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s > %d AND %s LIKE '%s%%'", tableName, tableName, colInt, minInt, colStr, prefix))

	// New: lots of IN query with non-consecutive 1000 elements for int, varchar, datetime
	{
		inCount := 1000
		// Int list
		intParts := []string{}
		if maxInt > minInt {
			rangeSize := maxInt - minInt
			step := rangeSize / inCount
			if step < 2 {
				step = 2
			}
			for v := minInt; v <= maxInt && len(intParts) < inCount; v += step {
				intParts = append(intParts, fmt.Sprintf("%d", v))
			}
		} else {
			intParts = append(intParts, fmt.Sprintf("%d", minInt))
		}
		// String list based on prefix + suffix range
		strParts := []string{}
		if suffixEnd >= suffixStart {
			rangeSize := suffixEnd - suffixStart
			sstep := rangeSize / inCount
			if sstep < 2 {
				sstep = 2
			}
			for v := suffixStart; v <= suffixEnd && len(strParts) < inCount; v += sstep {
				strParts = append(strParts, fmt.Sprintf("'%s%d'", prefix, v))
			}
			if len(strParts) == 0 {
				strParts = append(strParts, fmt.Sprintf("'%s%d'", prefix, suffixStart))
			}
		} else {
			strParts = append(strParts, fmt.Sprintf("'%s%d'", prefix, suffixStart))
		}
		// Datetime list from dateMin to dateMax
		dtParts := []string{}
		dtMax := dtMin
		if dateMax != "" {
			if t, err := time.Parse("2006-01-02", dateMax); err == nil {
				dtMax = t
			}
		}
		if !dtMax.Before(dtMin) {
			totalDays := int(dtMax.Sub(dtMin).Hours()/24) + 1
			dstep := totalDays / inCount
			if dstep < 2 {
				dstep = 2
			}
			for i := 0; i < totalDays && len(dtParts) < inCount; i += dstep {
				d := dtMin.AddDate(0, 0, i).Format("2006-01-02")
				dtParts = append(dtParts, fmt.Sprintf("'%s'", d))
			}
			if len(dtParts) == 0 {
				dtParts = append(dtParts, fmt.Sprintf("'%s'", dtMin.Format("2006-01-02")))
			}
		} else {
			dtParts = append(dtParts, fmt.Sprintf("'%s'", dtMin.Format("2006-01-02")))
		}
		ints := strings.Join(intParts, ", ")
		strs := strings.Join(strParts, ", ")
		dts := strings.Join(dtParts, ", ")
		sqls = append(sqls, fmt.Sprintf("/* LABEL: lots of IN */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s IN (%s) AND %s IN (%s) AND %s IN (%s)", tableName, tableName, colInt, ints, colStr, strs, colDt, dts))
	}
	content := strings.Join(sqls, ";\n") + ";\n"
	if oldContent != "" {
		content += oldContent
	}
	if err := os.WriteFile(outputFile, []byte(content), 0644); err != nil {
		fmt.Printf("Error writing query file: %v\n", err)
	} else {
		fmt.Printf("Generated queries file: %s\n", outputFile)
	}
}

// helpers duplicated to avoid importing from main
func parseIntValue(val interface{}) int {
	switch v := val.(type) {
	case int:
		return v
	case float64:
		return int(v)
	case string:
		var r int
		fmt.Sscanf(v, "%d", &r)
		return r
	default:
		return 0
	}
}
func getFloatValue(m map[string]interface{}, key string) float64 {
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
