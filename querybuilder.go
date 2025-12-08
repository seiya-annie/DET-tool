package main

import (
	"fmt"
	"os"
	"strings"
	"time"
)

// QueryBuilder builds SQL queries based on model configuration
type QueryBuilder struct{}

// NewQueryBuilder creates a new QueryBuilder
func NewQueryBuilder() *QueryBuilder {
	return &QueryBuilder{}
}

// Generate generates SQL queries based on model configuration and current stats
func (qb *QueryBuilder) Generate(modelConfig ModelConfig, tableName string, outputFile string, currentStats map[string]map[string]interface{}) {
	// Read existing content if file exists
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

	colInt := "col_int"
	colStr := "col_varchar"
	colDt := "col_datetime"

	// Get integer range from current stats or config
	var minInt, maxInt int
	if currentStats != nil {
		if intStats, ok := currentStats[colInt]; ok {
			if minVal, ok := intStats["min"]; ok && minVal != nil {
				minInt = parseIntValue(minVal)
			}
			if maxVal, ok := intStats["max"]; ok && maxVal != nil {
				maxInt = parseIntValue(maxVal)
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

	// Generate integer-based queries
	sqls = append(sqls, fmt.Sprintf("SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = %d", tableName, tableName, colInt, maxInt+1000))
	sqls = append(sqls, fmt.Sprintf("SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = %d", tableName, tableName, colInt, minInt+1))
	sqls = append(sqls, fmt.Sprintf("SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s BETWEEN %d AND %d", tableName, tableName, colInt, minInt, minInt+50))

	// Generate holes-specific queries if applicable
	if modelType == "holes" {
		if intHoleRange, ok := params["int_hole_range"].([]interface{}); ok && len(intHoleRange) >= 2 {
			holeStart := int(getFloatValue(map[string]interface{}{"val": intHoleRange[0]}, "val"))
			holeEnd := int(getFloatValue(map[string]interface{}{"val": intHoleRange[1]}, "val"))

			sqls = append(sqls, fmt.Sprintf("-- [Int] Holes Specific Queries"))
			sqls = append(sqls, fmt.Sprintf("SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s > %d AND %s < %d",
				tableName, tableName, colInt, holeStart, colInt, holeEnd))

			// Generate crossing query
			offset := (holeEnd - holeStart) / 10
			if offset < 500 {
				offset = 500
			}
			crossStart := minInt
			if holeStart-offset > minInt {
				crossStart = holeStart - offset
			}
			crossEnd := holeStart + offset
			sqls = append(sqls, fmt.Sprintf("SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s > %d AND %s < %d",
				tableName, tableName, colInt, crossStart, colInt, crossEnd))
		}
	}

	// Generate varchar-based queries
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

	sqls = append(sqls, fmt.Sprintf("SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = '%s%d'",
		tableName, tableName, colStr, prefix, suffixEnd+1000))
	sqls = append(sqls, fmt.Sprintf("SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = '%s%d'",
		tableName, tableName, colStr, prefix, suffixStart+1))
	sqls = append(sqls, fmt.Sprintf("SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s BETWEEN '%s%d' AND '%s%d'",
		tableName, tableName, colStr, prefix, suffixStart, prefix, suffixStart+50))

	// Generate datetime-based queries
	var dateMin, dateMax string
	if currentStats != nil {
		if dtStats, ok := currentStats[colDt]; ok {
			if maxVal, ok := dtStats["max"]; ok && maxVal != nil {
				dateMax = fmt.Sprintf("%v", maxVal)
			}
		}
	}

	if dateRange, ok := params["date_range"].([]interface{}); ok && len(dateRange) >= 2 {
		dateMin = fmt.Sprintf("%v", dateRange[0])
		if dateMax == "" {
			dateMax = fmt.Sprintf("%v", dateRange[1])
		}
	} else {
		dateMin = "2024-01-01"
		if dateMax == "" {
			dateMax = "2024-12-31"
		}
	}

	// Parse dates
	dtMin, err := time.Parse("2006-01-02", dateMin)
	if err != nil {
		dtMin = time.Now().AddDate(-1, 0, 0)
	}

	sqls = append(sqls, fmt.Sprintf("SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s > '%s'",
		tableName, tableName, colDt, dateMax))

	dtEq := dtMin.AddDate(0, 0, 1).Format("2006-01-02")
	sqls = append(sqls, fmt.Sprintf("SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = '%s'",
		tableName, tableName, colDt, dtEq))

	dtRangeEnd := dtMin.AddDate(0, 0, 30).Format("2006-01-02")
	sqls = append(sqls, fmt.Sprintf("SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s BETWEEN '%s' AND '%s'",
		tableName, tableName, colDt, dateMin, dtRangeEnd))

	// Generate datetime holes-specific queries if applicable
	if modelType == "holes" {
		if dateHoleRange, ok := params["date_hole_range"].([]interface{}); ok && len(dateHoleRange) >= 2 {
			dhStartStr := fmt.Sprintf("%v", dateHoleRange[0])
			dhEndStr := fmt.Sprintf("%v", dateHoleRange[1])

			sqls = append(sqls, "-- [Datetime] Holes Specific Queries")
			sqls = append(sqls, fmt.Sprintf("SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s > '%s' AND %s < '%s'",
				tableName, tableName, colDt, dhStartStr, colDt, dhEndStr))

			// Generate crossing query
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
				sqls = append(sqls, fmt.Sprintf("SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s > '%s' AND %s < '%s'",
					tableName, tableName, colDt, crossStart, colDt, crossEnd))
			}
		}
	}

	// Generate mixed condition query
	sqls = append(sqls, fmt.Sprintf("SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s > %d AND %s LIKE '%s%%'",
		tableName, tableName, colInt, minInt, colStr, prefix))

	// Write to file
	content := strings.Join(sqls, ";\n") + ";\n"
	if oldContent != "" {
		content += oldContent
	}

	writeErr := os.WriteFile(outputFile, []byte(content), 0644)
	if writeErr != nil {
		fmt.Printf("Error writing query file: %v\n", writeErr)
	} else {
		fmt.Printf("Generated queries file: %s\n", outputFile)
	}
}
