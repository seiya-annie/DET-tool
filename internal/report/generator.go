package report

import (
    "encoding/csv"
    "encoding/json"
    "fmt"
    "os"
    "sort"
    "strings"
    "time"

    itypes "det-tool/internal/types"
)

type ReportGenerator struct{}

func NewReportGenerator() *ReportGenerator { return &ReportGenerator{} }

// Sorting and helpers copied from existing reporter.go with type adjustments
func (rg *ReportGenerator) sortResults(results []itypes.QueryResult) []itypes.QueryResult {
    sorted := make([]itypes.QueryResult, len(results))
    copy(sorted, results)
    sort.Slice(sorted, func(i, j int) bool {
        isBadI := sorted[i].EstimationErrorRatio >= 10 && sorted[i].EstimationErrorValue >= 1000
        isBadJ := sorted[j].EstimationErrorRatio >= 10 && sorted[j].EstimationErrorValue >= 1000
        if isBadI != isBadJ { return isBadI }
        return sorted[i].EstimationErrorRatio > sorted[j].EstimationErrorRatio
    })
    return sorted
}

func (rg *ReportGenerator) GenerateCSVReport(results []itypes.QueryResult, filename string, config *itypes.Config, statsHealthy map[string]int, actual map[string]float64, useActual bool) error {
    file, err := os.Create(filename)
    if err != nil { return fmt.Errorf("failed to create CSV file: %v", err) }
    defer file.Close()

    w := csv.NewWriter(file)
    defer w.Flush()

    headers := []string{"Model", "Stats Healthy", "Modify Ratio", "Query Label", "Est Error Ratio", "Est Error Value", "Query SQL", "Duration (ms)", "Explain Plan", "Plan Replayer"}
    if err := w.Write(headers); err != nil { return fmt.Errorf("failed to write CSV header: %v", err) }

    sorted := rg.sortResults(results)
    for _, r := range sorted {
        healthyVal := rg.getStatsHealthyForModel(r.Model, statsHealthy)
        modifyRatio := rg.getModifyRatio(r.Model, config, actual, useActual)
        row := []string{
            r.Model,
            fmt.Sprintf("%d", healthyVal),
            fmt.Sprintf("%.3f", modifyRatio),
            r.QueryLabel,
            fmt.Sprintf("%.2f", r.EstimationErrorRatio),
            fmt.Sprintf("%.2f", r.EstimationErrorValue),
            r.Query,
            fmt.Sprintf("%.3f", r.DurationMs),
            r.Explain,
            r.PlanReplayerLink,
        }
        if err := w.Write(row); err != nil { return fmt.Errorf("failed to write CSV row: %v", err) }
    }
    fmt.Printf("CSV Report saved to: %s\n", filename)
    return nil
}

func (rg *ReportGenerator) GenerateHTMLReport(results []itypes.QueryResult, filename string, config *itypes.Config, statsHealthy map[string]int, actual map[string]float64, useActual bool) error {
    file, err := os.Create(filename)
    if err != nil { return fmt.Errorf("failed to create HTML file: %v", err) }
    defer file.Close()

    total := len(results)
    bad := 0
    for _, r := range results {
        if r.EstimationErrorRatio >= 10 && r.EstimationErrorValue >= 1000 { bad++ }
    }
    sorted := rg.sortResults(results)
    html := rg.generateHTMLContent(sorted, config, total, bad, statsHealthy, actual, useActual)
    if _, err := file.WriteString(html); err != nil { return fmt.Errorf("failed to write HTML content: %v", err) }
    fmt.Printf("HTML Report saved to: %s\n", filename)
    return nil
}

func (rg *ReportGenerator) GenerateJSONReport(results []itypes.QueryResult, filename string, config *itypes.Config, statsHealthy map[string]int, actual map[string]float64, useActual bool) error {
    type Extended struct {
        itypes.QueryResult
        StatsHealthy int     `json:"stats_healthy"`
        ModifyRatio  float64 `json:"modify_ratio"`
        IsBadCase    bool    `json:"is_bad_case"`
    }
    sorted := rg.sortResults(results)
    ext := make([]Extended, len(sorted))
    for i, r := range sorted {
        ext[i] = Extended{QueryResult: r, StatsHealthy: rg.getStatsHealthyForModel(r.Model, statsHealthy), ModifyRatio: rg.getModifyRatio(r.Model, config, actual, useActual), IsBadCase: r.EstimationErrorRatio >= 10 && r.EstimationErrorValue >= 1000}
    }
    report := struct {
        GeneratedAt   time.Time           `json:"generated_at"`
        TotalQueries  int                 `json:"total_queries"`
        Results       []Extended          `json:"results"`
        Summary       map[string]interface{} `json:"summary"`
        Configuration *itypes.Config      `json:"configuration"`
    }{GeneratedAt: time.Now(), TotalQueries: len(results), Results: ext, Configuration: config}
    report.Summary = rg.calculateSummaryStats(results)
    data, err := json.MarshalIndent(report, "", "  ")
    if err != nil { return fmt.Errorf("failed to marshal JSON: %v", err) }
    if err := os.WriteFile(filename, data, 0644); err != nil { return fmt.Errorf("failed to write JSON file: %v", err) }
    fmt.Printf("JSON Report saved to: %s\n", filename)
    return nil
}

// Helpers below are adapted from existing reporter.go
func (rg *ReportGenerator) getStatsHealthyForModel(modelName string, statsHealthy map[string]int) int {
    if val, ok := statsHealthy[modelName]; ok { return val }
    return 100
}

func (rg *ReportGenerator) calculateModifyRatio(modelName string, config *itypes.Config) float64 {
    for _, m := range config.Models {
        if m.Name == modelName {
            params := m.Params
            inc := m.Incremental
            if inc == nil { return 0 }
            baseRows := getFloat(params, "rows")
            if baseRows == 0 { baseRows = 1000 }
            insertRows := getFloat(inc, "insert_rows")
            updateRatio := getFloat(inc, "update_ratio")
            deleteRatio := getFloat(inc, "delete_ratio")
            return (insertRows / baseRows) + updateRatio + deleteRatio
        }
    }
    return 0
}

// getModifyRatio returns actual ratio if enabled and present; otherwise target ratio from config
func (rg *ReportGenerator) getModifyRatio(modelName string, config *itypes.Config, actual map[string]float64, useActual bool) float64 {
    if useActual && actual != nil {
        if v, ok := actual[modelName]; ok {
            return v
        }
    }
    return rg.calculateModifyRatio(modelName, config)
}

func (rg *ReportGenerator) calculateSummaryStats(results []itypes.QueryResult) map[string]interface{} {
    if len(results) == 0 {
        return map[string]interface{}{"total_queries": 0, "bad_cases": 0, "success_rate": 100.0}
    }
    total := len(results)
    bad := 0
    var totalRatio, totalValue, totalDur float64
    for _, r := range results {
        if r.EstimationErrorRatio >= 10 && r.EstimationErrorValue >= 1000 { bad++ }
        totalRatio += r.EstimationErrorRatio
        totalValue += r.EstimationErrorValue
        totalDur += r.DurationMs
    }
    success := float64(total-bad) / float64(total) * 100
    return map[string]interface{}{
        "total_queries": total,
        "bad_cases": bad,
        "success_rate": success,
        "avg_estimation_error_ratio": totalRatio / float64(total),
        "avg_estimation_error_value": totalValue / float64(total),
        "avg_duration_ms": totalDur / float64(total),
    }
}

func (rg *ReportGenerator) DisplayTopQueries(results []itypes.QueryResult, count int) {
    if count <= 0 { count = 10 }
    if len(results) == 0 { fmt.Println("No query results to display."); return }
    sorted := rg.sortResults(results)
    fmt.Printf("\n=== Top %d Queries by Estimation Error Ratio ===\n", count)
    fmt.Printf("%-4s %-20s %-15s %-15s %-12s %s\n", "Rank", "Model", "Error Ratio", "Error Value", "Duration(ms)", "Query")
    fmt.Println(strings.Repeat("-", 120))
    for i := 0; i < count && i < len(sorted); i++ {
        r := sorted[i]
        q := r.Query
        if len(q) > 80 { q = q[:77] + "..." }
        fmt.Printf("%-4d %-20s %-15.2f %-15.2f %-12.3f %s\n", i+1, r.Model, r.EstimationErrorRatio, r.EstimationErrorValue, r.DurationMs, q)
    }
}

// minimal helpers replacing utils.getFloatValue
func getFloat(m map[string]interface{}, key string) float64 {
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
    return 0
}

func escapeHTML(s string) string { return strings.ReplaceAll(strings.ReplaceAll(s, "<", "&lt;"), ">", "&gt;") }

// truncateString shortens a string to maxLen with ellipsis
func truncateString(s string, maxLen int) string {
    if maxLen <= 3 || len(s) <= maxLen { return s }
    return s[:maxLen-3] + "..."
}

func (rg *ReportGenerator) generateHTMLContent(results []itypes.QueryResult, config *itypes.Config, totalQueries, badCases int, statsHealthy map[string]int, actual map[string]float64, useActual bool) string {
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
        .label-cell {
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            white-space: normal;
            word-wrap: break-word;
            word-break: break-word;
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
                    <th>Plan Replayer</th>
                </tr>
            </thead>
            <tbody>
%s
            </tbody>
        </table>
    </div>
</body>
</html>`

    // Generate model statistics and rows using helpers mirroring original template
    modelStats := rg.generateModelStats(results, config, actual, useActual)
    tableRows := rg.generateTableRows(results, config, statsHealthy, actual, useActual)
    successRate := 100.0
    if totalQueries > 0 {
        successRate = float64(totalQueries-badCases) / float64(totalQueries) * 100.0
    }
    return fmt.Sprintf(htmlTemplate, currentTime, currentTime, totalQueries, totalQueries, badCases, successRate, modelStats, tableRows)
}

// generateModelStats generates statistics by model (preserving original style)
func (rg *ReportGenerator) generateModelStats(results []itypes.QueryResult, config *itypes.Config, actual map[string]float64, useActual bool) string {
    modelStats := make(map[string]struct {
        TotalQueries  int
        BadCases      int
        AvgErrorRatio float64
        AvgErrorValue float64
        AvgDuration   float64
    })
    for _, r := range results {
        st := modelStats[r.Model]
        st.TotalQueries++
        if r.EstimationErrorRatio >= 10 && r.EstimationErrorValue >= 1000 { st.BadCases++ }
        st.AvgErrorRatio += r.EstimationErrorRatio
        st.AvgErrorValue += r.EstimationErrorValue
        st.AvgDuration += r.DurationMs
        modelStats[r.Model] = st
    }
    var b strings.Builder
    for model, st := range modelStats {
        if st.TotalQueries > 0 {
            st.AvgErrorRatio /= float64(st.TotalQueries)
            st.AvgErrorValue /= float64(st.TotalQueries)
            st.AvgDuration /= float64(st.TotalQueries)
        }
        b.WriteString(fmt.Sprintf(`
        <div class="model-stats">
            <strong>%s</strong><br>
            Total Queries: %d | Bad Cases: %d | Avg Error Ratio: %.2f | Avg Duration: %.2fms
        </div>`, model, st.TotalQueries, st.BadCases, st.AvgErrorRatio, st.AvgDuration))
    }
    return b.String()
}

// generateTableRows generates HTML table rows (preserving original style)
func (rg *ReportGenerator) generateTableRows(results []itypes.QueryResult, config *itypes.Config, statsHealthy map[string]int, actual map[string]float64, useActual bool) string {
    var b strings.Builder
    for i, r := range results {
        if i >= 500 { break }
        isRiskQuery := r.EstimationErrorRatio >= 10 && r.EstimationErrorValue >= 1000
        rowClass := ""
        if isRiskQuery { rowClass = `class="high-error"` }
        healthyVal := rg.getStatsHealthyForModel(r.Model, statsHealthy)
        modifyRatio := rg.getModifyRatio(r.Model, config, actual, useActual)
        b.WriteString(fmt.Sprintf(`
                <tr %s>
                    <td>%s</td>
                    <td class="numeric-cell">%d</td>
                    <td class="numeric-cell">%.3f</td>
                    <td class="label-cell">%s</td>
                    <td class="numeric-cell">%.2f</td>
                    <td class="numeric-cell">%.2f</td>
                    <td class="query-cell">%s</td>
                    <td class="numeric-cell">%.3f</td>
                    <td class="explain-cell">%s</td>
                    <td class="numeric-cell">%s</td>
                </tr>`,
            rowClass,
            escapeHTML(r.Model),
            healthyVal,
            modifyRatio,
            escapeHTML(r.QueryLabel),
            r.EstimationErrorRatio,
            r.EstimationErrorValue,
            escapeHTML(truncateString(r.Query, 200)),
            r.DurationMs,
            escapeHTML(r.Explain),
            escapeHTML(r.PlanReplayerLink),
        ))
    }
    return b.String()
}
