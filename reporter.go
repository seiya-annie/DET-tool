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

// GenerateCSVReport generates a CSV report from query results
func (rg *ReportGenerator) GenerateCSVReport(results []QueryResult, filename string, config *Config) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create CSV file: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	headers := []string{
		"Model", "stats_healthy_ratio", "modify_ratio", "query_label",
		"estimation_error_ratio", "estimation_error_value", "query",
		"duration_ms", "explain", "risk_operators_count",
	}
	
	if err := writer.Write(headers); err != nil {
		return fmt.Errorf("failed to write CSV header: %v", err)
	}

	// Write data rows
	for _, result := range results {
		statsHealthy := rg.getStatsHealthyForModel(result.Model, config)
		modifyRatio := rg.calculateModifyRatio(result.Model, config)

		row := []string{
			result.Model,
			fmt.Sprintf("%.3f", statsHealthy),
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
func (rg *ReportGenerator) GenerateHTMLReport(results []QueryResult, filename string, config *Config) error {
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

	// Sort results by estimation error ratio (descending)
	sortedResults := make([]QueryResult, len(results))
	copy(sortedResults, results)
	sort.Slice(sortedResults, func(i, j int) bool {
		return sortedResults[i].EstimationErrorRatio > sortedResults[j].EstimationErrorRatio
	})

	// Generate HTML content
	htmlContent := rg.generateHTMLContent(sortedResults, config, totalQueries, badCases)

	if _, err := file.WriteString(htmlContent); err != nil {
		return fmt.Errorf("failed to write HTML content: %v", err)
	}

	fmt.Printf("HTML Report saved to: %s\n", filename)
	return nil
}

// generateHTMLContent generates the HTML content for the report
func (rg *ReportGenerator) generateHTMLContent(results []QueryResult, config *Config, totalQueries, badCases int) string {
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
            max-width: 1400px; 
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
            max-width: 300px; 
            word-wrap: break-word; 
            font-family: 'Courier New', monospace; 
            font-size: 11px; 
        }
        .explain-cell { 
            max-width: 400px; 
            word-wrap: break-word; 
            font-family: 'Courier New', monospace; 
            font-size: 10px; 
            color: #666; 
            white-space: pre-wrap; 
        }
        .numeric-cell { 
            text-align: left; 
            font-family: 'Courier New', monospace; 
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
        .chart-container {
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
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
                    <th>Stats Healthy Ratio</th>
                    <th>Modify Ratio</th>
                    <th>Query Label</th>
                    <th>Estimation Error Ratio</th>
                    <th>Estimation Error Value</th>
                    <th>Query SQL</th>
                    <th>Duration (ms)</th>
                    <th>Explain Plan</th>
                    <th>Risk Operators</th>
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
	tableRows := rg.generateTableRows(results, config)

	successRate := 100.0
	if totalQueries > 0 {
		successRate = float64(totalQueries-badCases) / float64(totalQueries) * 100
	}

	return fmt.Sprintf(htmlTemplate,
		currentTime,
		currentTime,
		totalQueries,
		badCases,
		successRate,
		modelStats,
		tableRows)
}

// generateModelStats generates statistics by model
func (rg *ReportGenerator) generateModelStats(results []QueryResult, config *Config) string {
	modelStats := make(map[string]struct {
		TotalQueries     int
		BadCases         int
		AvgErrorRatio    float64
		AvgErrorValue    float64
		AvgDuration      float64
	})

	// Aggregate statistics by model
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

	// Calculate averages
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
func (rg *ReportGenerator) generateTableRows(results []QueryResult, config *Config) string {
	var html strings.Builder
	
	for i, result := range results {
		// Limit to top 100 results
		if i >= 100 {
			break
		}

		isRiskQuery := result.EstimationErrorRatio >= 10 && result.EstimationErrorValue >= 1000
		rowClass := ""
		if isRiskQuery {
			rowClass = `class="high-error"`
		}

		statsHealthy := rg.getStatsHealthyForModel(result.Model, config)
		modifyRatio := rg.calculateModifyRatio(result.Model, config)

		html.WriteString(fmt.Sprintf(`
                <tr %s>
                    <td>%s</td>
                    <td class="numeric-cell">%.3f</td>
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
			statsHealthy,
			modifyRatio,
			"",
			result.EstimationErrorRatio,
			result.EstimationErrorValue,
			escapeHTML(truncateString(result.Query, 200)),
			result.DurationMs,
			escapeHTML(truncateString(result.Explain, 500)),
			result.RiskOperatorsCount,
		))
	}

	return html.String()
}

// getStatsHealthyForModel gets stats healthy ratio for a model
func (rg *ReportGenerator) getStatsHealthyForModel(modelName string, config *Config) float64 {
	// This is a simplified implementation
	// In a real implementation, you would get this from the database
	return 1.0
}

// calculateModifyRatio calculates the modify ratio for a model
func (rg *ReportGenerator) calculateModifyRatio(modelName string, config *Config) float64 {
	for _, model := range config.Models {
		if model.Name == modelName {
			incremental := model.Incremental
			if incremental == nil {
				return 0.0
			}

			baseRows := getFloatValue(model.Params, "rows")
			insertRows := getFloatValue(incremental, "insert_rows")
			updateRatio := getFloatValue(incremental, "update_ratio")
			deleteRatio := getFloatValue(incremental, "delete_ratio")

			if baseRows > 0 {
				return (insertRows/baseRows) + updateRatio + deleteRatio
			}
		}
	}
	return 0.0
}

// GenerateJSONReport generates a JSON report from query results
func (rg *ReportGenerator) GenerateJSONReport(results []QueryResult, filename string, config *Config) error {
	report := struct {
		GeneratedAt   time.Time              `json:"generated_at"`
		TotalQueries  int                    `json:"total_queries"`
		Results       []QueryResult          `json:"results"`
		Summary       map[string]interface{} `json:"summary"`
		Configuration *Config                `json:"configuration"`
	}{
		GeneratedAt:   time.Now(),
		TotalQueries:  len(results),
		Results:       results,
		Configuration: config,
	}

	// Calculate summary statistics
	summary := rg.calculateSummaryStats(results)
	report.Summary = summary

	// Marshal to JSON
	jsonData, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %v", err)
	}

	// Write to file
	if err := os.WriteFile(filename, jsonData, 0644); err != nil {
		return fmt.Errorf("failed to write JSON file: %v", err)
	}

	fmt.Printf("JSON Report saved to: %s\n", filename)
	return nil
}

// calculateSummaryStats calculates summary statistics
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
		"total_queries":           totalQueries,
		"bad_cases":               badCases,
		"success_rate":            successRate,
		"avg_estimation_error_ratio": totalErrorRatio / float64(totalQueries),
		"avg_estimation_error_value": totalErrorValue / float64(totalQueries),
		"avg_duration_ms":         totalDuration / float64(totalQueries),
	}
}

// DisplayTopQueries displays top queries by estimation error
func (rg *ReportGenerator) DisplayTopQueries(results []QueryResult, count int) {
	if count <= 0 {
		count = 10
	}

	if len(results) == 0 {
		fmt.Println("No query results to display.")
		return
	}

	// Sort by estimation error ratio
	sorted := make([]QueryResult, len(results))
	copy(sorted, results)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].EstimationErrorRatio > sorted[j].EstimationErrorRatio
	})

	// Display header
	fmt.Printf("\n=== Top %d Queries by Estimation Error Ratio ===\n", count)
	fmt.Printf("%-4s %-20s %-15s %-15s %-12s %s\n", "Rank", "Model", "Error Ratio", "Error Value", "Duration(ms)", "Query")
	fmt.Println(strings.Repeat("-", 120))

	// Display top queries
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

// Helper functions



// ExportResultsForAnalysis exports results in a format suitable for further analysis
func (rg *ReportGenerator) ExportResultsForAnalysis(results []QueryResult, filename string, config *Config) error {
	// Create a more detailed export for analysis
	type AnalysisRecord struct {
		Timestamp              string  `json:"timestamp"`
		Model                  string  `json:"model"`
		QueryID                int     `json:"query_id"`
		Query                  string  `json:"query"`
		DurationMs             float64 `json:"duration_ms"`
		EstimationErrorValue   float64 `json:"estimation_error_value"`
		EstimationErrorRatio   float64 `json:"estimation_error_ratio"`
		RiskOperatorsCount     int     `json:"risk_operators_count"`
		IsRiskQuery            bool    `json:"is_risk_query"`
		StatsHealthyRatio      float64 `json:"stats_healthy_ratio"`
		ModifyRatio            float64 `json:"modify_ratio"`
	}

	records := make([]AnalysisRecord, len(results))
	for i, result := range results {
		records[i] = AnalysisRecord{
			Timestamp:            time.Now().Format("2006-01-02 15:04:05"),
			Model:                result.Model,
			QueryID:              result.QueryID,
			Query:                result.Query,
			DurationMs:           result.DurationMs,
			EstimationErrorValue: result.EstimationErrorValue,
			EstimationErrorRatio: result.EstimationErrorRatio,
			RiskOperatorsCount:   result.RiskOperatorsCount,
			IsRiskQuery:          result.EstimationErrorRatio >= 10 && result.EstimationErrorValue >= 1000,
			StatsHealthyRatio:    rg.getStatsHealthyForModel(result.Model, config),
			ModifyRatio:          rg.calculateModifyRatio(result.Model, config),
		}
	}

	jsonData, err := json.MarshalIndent(records, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal analysis data: %v", err)
	}

	if err := os.WriteFile(filename, jsonData, 0644); err != nil {
		return fmt.Errorf("failed to write analysis file: %v", err)
	}

	fmt.Printf("Analysis export saved to: %s\n", filename)
	return nil
}