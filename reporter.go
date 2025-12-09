package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"
)

// ReportGenerator generates reports from query results
type ReportGenerator struct{}

// NewReportGenerator creates a new ReportGenerator
func NewReportGenerator() *ReportGenerator {
	return &ReportGenerator{}
}

// Helper to sort results by EstimationErrorRatio descending
func (rg *ReportGenerator) sortResults(results []QueryResult) []QueryResult {
	sorted := make([]QueryResult, len(results))
	copy(sorted, results)
	sort.Slice(sorted, func(i, j int) bool {
		// 降序排列: Ratio 大的在前
		return sorted[i].EstimationErrorRatio > sorted[j].EstimationErrorRatio
	})
	return sorted
}

// GenerateCSVReport generates a CSV report from query results
func (rg *ReportGenerator) GenerateCSVReport(results []QueryResult, filename string, config *Config, statsHealthy map[string]int) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create CSV file: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	headers := []string{
		"Model", "Stats Healthy", "Modify Ratio", "Query Label",
		"Est Error Ratio", "Est Error Value", "Query SQL", "Duration (ms)",
		"Explain Plan", "Risk Operators Count",
	}

	if err := writer.Write(headers); err != nil {
		return fmt.Errorf("failed to write CSV header: %v", err)
	}

	// [修改] 排序结果
	sortedResults := rg.sortResults(results)

	// Write data rows
	for _, result := range sortedResults {
		healthyVal := rg.getStatsHealthyForModel(result.Model, statsHealthy)
		modifyRatio := rg.calculateModifyRatio(result.Model, config)

		row := []string{
			result.Model,
			fmt.Sprintf("%d", healthyVal),
			fmt.Sprintf("%.3f", modifyRatio),
			"", // query_label
			fmt.Sprintf("%.2f", result.EstimationErrorRatio),
			fmt.Sprintf("%.2f", result.EstimationErrorValue),
			result.Query,
			fmt.Sprintf("%.3f", result.DurationMs),
			result.Explain,
			fmt.Sprintf("%d", result.RiskOperatorsCount),
		}

		if err := writer.Write(row); err != nil {
			return fmt.Errorf("failed to write CSV row: %v", err)
		}
	}

	fmt.Printf("CSV Report saved to: %s\n", filename)
	return nil
}

// GenerateHTMLReport generates an HTML report from query results
func (rg *ReportGenerator) GenerateHTMLReport(results []QueryResult, filename string, config *Config, statsHealthy map[string]int) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create HTML file: %v", err)
	}
	defer file.Close()

	// Calculate statistics
	totalQueries := len(results)
	badCases := 0
	for _, result := range results {
		if result.EstimationErrorRatio >= 10 && result.EstimationErrorValue >= 1000 {
			badCases++
		}
	}

	// [修改] 使用统一的排序函数
	sortedResults := rg.sortResults(results)

	// Generate HTML content
	htmlContent := rg.generateHTMLContent(sortedResults, config, totalQueries, badCases, statsHealthy)

	if _, err := file.WriteString(htmlContent); err != nil {
		return fmt.Errorf("failed to write HTML content: %v", err)
	}

	fmt.Printf("HTML Report saved to: %s\n", filename)
	return nil
}

// generateHTMLContent generates the HTML content for the report
func (rg *ReportGenerator) generateHTMLContent(results []QueryResult, config *Config, totalQueries, badCases int, statsHealthy map[string]int) string {
	currentTime := time.Now().Format("2006-01-02 15:04:05")

	htmlTemplate := `<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Query Execution Report - %s</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5; 
        }
        .container { 
            width: 95%%; 
            max-width: 1800px; 
            margin: 0 auto; 
            background-color: white; 
            padding: 20px; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        }
        h1 { 
            color: #333; 
            text-align: center; 
            margin-bottom: 30px; 
        }
        .summary { 
            background-color: #e8f4f8; 
            padding: 15px; 
            border-radius: 5px; 
            margin-bottom: 20px; 
        }
        table { 
            width: 100%%; 
            border-collapse: collapse; 
            margin-top: 20px; 
            font-size: 12px; 
        }
        th { 
            background-color: #4CAF50; 
            color: white; 
            padding: 12px 8px; 
            text-align: left; 
            font-weight: bold; 
            position: sticky; 
            top: 0; 
            z-index: 10; 
            white-space: nowrap; 
        }
        td { 
            padding: 8px; 
            border-bottom: 1px solid #ddd; 
            vertical-align: top; 
        }
        tr:nth-child(even) { 
            background-color: #f9f9f9; 
        }
        tr:hover { 
            background-color: #f0f0f0; 
        }
        .high-error { 
            background-color: #ffebee !important; 
            color: #c62828; 
            font-weight: bold; 
        }
        .query-cell { 
            width: 20%%;          
            word-wrap: break-word; 
            word-break: break-all; 
            font-family: 'Consolas', 'Monaco', monospace; 
            font-size: 11px; 
        }
        .explain-cell { 
            width: 40%%;           
            min-width: 400px;      
            max-width: 800px;      
            font-family: 'Consolas', 'Monaco', monospace; 
            font-size: 11px; 
            color: #333; 
            white-space: pre-wrap; 
            word-wrap: break-word; 
            word-break: break-all; 
            background-color: #f8f9fa;
            padding: 8px;
            border: 1px solid #eee;
            border-radius: 4px;
        }
        .numeric-cell { 
            text-align: left; 
            font-family: 'Consolas', 'Monaco', monospace; 
        }
        .stats { 
            display: flex; 
            justify-content: space-around; 
            margin-bottom: 20px; 
        }
        .stat-box { 
            background-color: #f8f9fa; 
            padding: 15px; 
            border-radius: 5px; 
            text-align: center; 
            min-width: 120px; 
        }
        .stat-value { 
            font-size: 24px; 
            font-weight: bold; 
            color: #4CAF50; 
        }
        .stat-label { 
            font-size: 12px; 
            color: #666; 
            margin-top: 5px; 
        }
        .model-stats {
            margin: 10px 0;
            padding: 10px;
            background-color: #e8f4f8;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Query Execution Analysis Report</h1>
        <div class="summary">
            <strong>Report Generated:</strong> %s<br>
            <strong>Total Queries:</strong> %d
        </div>

        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">%d</div>
                <div class="stat-label">Total Queries</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">%d</div>
                <div class="stat-label">Bad Cases<br>(Risk Queries)</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">%.1f%%</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>

        %s

        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Stats Healthy</th>
                    <th>Modify Ratio</th>
                    <th>Query Label</th>
                    <th>Est Error Ratio</th>
                    <th>Est Error Value</th>
                    <th>Query SQL</th>
                    <th>Duration (ms)</th>
                    <th>Explain Plan</th>
                    <th>Risk Ops</th>
                </tr>
            </thead>
            <tbody>
%s
            </tbody>
        </table>
    </div>
</body>
</html>`

	// Generate model statistics
	modelStats := rg.generateModelStats(results, config)

	// Generate table rows
	tableRows := rg.generateTableRows(results, config, statsHealthy)

	successRate := 100.0
	if totalQueries > 0 {
		successRate = float64(totalQueries-badCases) / float64(totalQueries) * 100.0
	}

	return fmt.Sprintf(htmlTemplate,
		currentTime,
		currentTime,
		totalQueries,
		totalQueries,
		badCases,
		successRate,
		modelStats,
		tableRows)
}

// generateModelStats generates statistics by model
func (rg *ReportGenerator) generateModelStats(results []QueryResult, config *Config) string {
	modelStats := make(map[string]struct {
		TotalQueries  int
		BadCases      int
		AvgErrorRatio float64
		AvgErrorValue float64
		AvgDuration   float64
	})

	for _, result := range results {
		stats := modelStats[result.Model]
		stats.TotalQueries++
		if result.EstimationErrorRatio >= 10 && result.EstimationErrorValue >= 1000 {
			stats.BadCases++
		}
		stats.AvgErrorRatio += result.EstimationErrorRatio
		stats.AvgErrorValue += result.EstimationErrorValue
		stats.AvgDuration += result.DurationMs
		modelStats[result.Model] = stats
	}

	var html strings.Builder
	for model, stats := range modelStats {
		if stats.TotalQueries > 0 {
			stats.AvgErrorRatio /= float64(stats.TotalQueries)
			stats.AvgErrorValue /= float64(stats.TotalQueries)
			stats.AvgDuration /= float64(stats.TotalQueries)
		}

		html.WriteString(fmt.Sprintf(`
        <div class="model-stats">
            <strong>%s</strong><br>
            Total Queries: %d | Bad Cases: %d | Avg Error Ratio: %.2f | Avg Duration: %.2fms
        </div>`,
			model, stats.TotalQueries, stats.BadCases, stats.AvgErrorRatio, stats.AvgDuration))
	}

	return html.String()
}

// generateTableRows generates HTML table rows
func (rg *ReportGenerator) generateTableRows(results []QueryResult, config *Config, statsHealthy map[string]int) string {
	var html strings.Builder

	for i, result := range results {
		if i >= 500 {
			break
		}

		isRiskQuery := result.EstimationErrorRatio >= 10 && result.EstimationErrorValue >= 1000
		rowClass := ""
		if isRiskQuery {
			rowClass = `class="high-error"`
		}

		healthyVal := rg.getStatsHealthyForModel(result.Model, statsHealthy)
		modifyRatio := rg.calculateModifyRatio(result.Model, config)

		html.WriteString(fmt.Sprintf(`
                <tr %s>
                    <td>%s</td>
                    <td class="numeric-cell">%d</td>
                    <td class="numeric-cell">%.3f</td>
                    <td>%s</td>
                    <td class="numeric-cell">%.2f</td>
                    <td class="numeric-cell">%.2f</td>
                    <td class="query-cell">%s</td>
                    <td class="numeric-cell">%.3f</td>
                    <td class="explain-cell">%s</td>
                    <td class="numeric-cell">%d</td>
                </tr>`,
			rowClass,
			escapeHTML(result.Model),
			healthyVal,
			modifyRatio,
			"",
			result.EstimationErrorRatio,
			result.EstimationErrorValue,
			escapeHTML(truncateString(result.Query, 200)),
			result.DurationMs,
			escapeHTML(result.Explain),
			result.RiskOperatorsCount,
		))
	}

	return html.String()
}

// getStatsHealthyForModel gets stats healthy value for a model
func (rg *ReportGenerator) getStatsHealthyForModel(modelName string, statsHealthy map[string]int) int {
	if val, ok := statsHealthy[modelName]; ok {
		return val
	}
	return 100
}

// calculateModifyRatio calculates the modify ratio for a model
func (rg *ReportGenerator) calculateModifyRatio(modelName string, config *Config) float64 {
	for _, model := range config.Models {
		if model.Name == modelName {
			params := model.Params
			incremental := model.Incremental
			if incremental == nil {
				return 0.0
			}

			baseRows := getFloatValue(params, "rows")
			if baseRows == 0 {
				baseRows = 1000
			}

			insertRows := getFloatValue(incremental, "insert_rows")
			updateRatio := getFloatValue(incremental, "update_ratio")
			deleteRatio := getFloatValue(incremental, "delete_ratio")

			return (insertRows / baseRows) + updateRatio + deleteRatio
		}
	}
	return 0.0
}

// GenerateJSONReport generates a JSON report from query results
func (rg *ReportGenerator) GenerateJSONReport(results []QueryResult, filename string, config *Config, statsHealthy map[string]int) error {
	type ExtendedQueryResult struct {
		QueryResult
		StatsHealthy int     `json:"stats_healthy"`
		ModifyRatio  float64 `json:"modify_ratio"`
		IsBadCase    bool    `json:"is_bad_case"`
	}

	// [修改] 排序结果
	sortedResults := rg.sortResults(results)

	extendedResults := make([]ExtendedQueryResult, len(sortedResults))
	for i, r := range sortedResults {
		extendedResults[i] = ExtendedQueryResult{
			QueryResult:  r,
			StatsHealthy: rg.getStatsHealthyForModel(r.Model, statsHealthy),
			ModifyRatio:  rg.calculateModifyRatio(r.Model, config),
			IsBadCase:    r.EstimationErrorRatio >= 10 && r.EstimationErrorValue >= 1000,
		}
	}

	report := struct {
		GeneratedAt   time.Time              `json:"generated_at"`
		TotalQueries  int                    `json:"total_queries"`
		Results       []ExtendedQueryResult  `json:"results"`
		Summary       map[string]interface{} `json:"summary"`
		Configuration *Config                `json:"configuration"`
	}{
		GeneratedAt:   time.Now(),
		TotalQueries:  len(results),
		Results:       extendedResults,
		Configuration: config,
	}

	summary := rg.calculateSummaryStats(results)
	report.Summary = summary

	jsonData, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %v", err)
	}

	if err := os.WriteFile(filename, jsonData, 0644); err != nil {
		return fmt.Errorf("failed to write JSON file: %v", err)
	}

	fmt.Printf("JSON Report saved to: %s\n", filename)
	return nil
}

func (rg *ReportGenerator) calculateSummaryStats(results []QueryResult) map[string]interface{} {
	if len(results) == 0 {
		return map[string]interface{}{
			"total_queries": 0,
			"bad_cases":     0,
			"success_rate":  100.0,
		}
	}

	totalQueries := len(results)
	badCases := 0
	var totalErrorRatio, totalErrorValue, totalDuration float64

	for _, result := range results {
		if result.EstimationErrorRatio >= 10 && result.EstimationErrorValue >= 1000 {
			badCases++
		}
		totalErrorRatio += result.EstimationErrorRatio
		totalErrorValue += result.EstimationErrorValue
		totalDuration += result.DurationMs
	}

	successRate := float64(totalQueries-badCases) / float64(totalQueries) * 100

	return map[string]interface{}{
		"total_queries":              totalQueries,
		"bad_cases":                  badCases,
		"success_rate":               successRate,
		"avg_estimation_error_ratio": totalErrorRatio / float64(totalQueries),
		"avg_estimation_error_value": totalErrorValue / float64(totalQueries),
		"avg_duration_ms":            totalDuration / float64(totalQueries),
	}
}

func (rg *ReportGenerator) DisplayTopQueries(results []QueryResult, count int) {
	if count <= 0 {
		count = 10
	}

	if len(results) == 0 {
		fmt.Println("No query results to display.")
		return
	}

	// [修改] 使用统一的排序函数
	sorted := rg.sortResults(results)

	fmt.Printf("\n=== Top %d Queries by Estimation Error Ratio ===\n", count)
	fmt.Printf("%-4s %-20s %-15s %-15s %-12s %s\n", "Rank", "Model", "Error Ratio", "Error Value", "Duration(ms)", "Query")
	fmt.Println(strings.Repeat("-", 120))

	for i := 0; i < count && i < len(sorted); i++ {
		result := sorted[i]
		query := result.Query
		if len(query) > 80 {
			query = query[:77] + "..."
		}

		fmt.Printf("%-4d %-20s %-15.2f %-15.2f %-12.3f %s\n",
			i+1,
			result.Model,
			result.EstimationErrorRatio,
			result.EstimationErrorValue,
			result.DurationMs,
			query,
		)
	}
}
