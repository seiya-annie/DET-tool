package main

import (
    "encoding/csv"
    "encoding/json"
    "flag"
    "fmt"
    "io/fs"
    "os"
    "os/exec"
    "path/filepath"
    "sort"
    "strings"
    "time"

    itypes "det-tool/internal/types"
)

// QueryRow represents a single query record aggregated across scenarios.
// Defined at package level so it can be shared between helpers and main.
type QueryRow struct {
    Label        string
    Model        string
    StatsHealthy int
    ModifyRatio  float64
    QueryLabel   string
    EstErrRatio  float64
    EstErrValue  float64
    QuerySQL     string
}

func readJSON(path string, v interface{}) error {
    b, err := os.ReadFile(path)
    if err != nil {
        return err
    }
    return json.Unmarshal(b, v)
}

func writeJSON(path string, v interface{}) error {
    b, err := json.MarshalIndent(v, "", "  ")
    if err != nil {
        return err
    }
    if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
        return err
    }
    return os.WriteFile(path, b, 0644)
}

func getFloat(m map[string]interface{}, key string) float64 {
    if m == nil { return 0 }
    if v, ok := m[key]; ok {
        switch x := v.(type) {
        case float64:
            return x
        case float32:
            return float64(x)
        case int:
            return float64(x)
        case int64:
            return float64(x)
        case string:
            var f float64
            fmt.Sscanf(x, "%f", &f)
            return f
        }
    }
    return 0
}

func roundInt(x float64) int { if x < 0 { return 0 }; return int(x + 0.5) }

func adjustConfigForRatio(base itypes.Config, target float64, includeExternal bool) itypes.Config {
    out := itypes.Config{Models: make([]itypes.ModelConfig, 0, len(base.Models))}
    for _, m := range base.Models {
        if !includeExternal {
            if strings.HasPrefix(m.Type, "external_") {
                continue
            }
        }
        mc := itypes.ModelConfig{
            Name:        m.Name,
            Description: m.Description,
            Type:        m.Type,
            Params:      map[string]interface{}{},
            Incremental: map[string]interface{}{},
        }
        for k, v := range m.Params { mc.Params[k] = v }
        for k, v := range m.Incremental { mc.Incremental[k] = v }

        rows := getFloat(mc.Params, "rows")
        if rows <= 0 { rows = 1000 }

        baseInsert := getFloat(mc.Incremental, "insert_rows") / rows
        baseUpdate := getFloat(mc.Incremental, "update_ratio")
        baseDelete := getFloat(mc.Incremental, "delete_ratio")
        baseSum := baseInsert + baseUpdate + baseDelete

        var up, del, insFrac float64
        if baseSum > 0 {
            scale := target / baseSum
            up = baseUpdate * scale
            del = baseDelete * scale
            if up > 0.8 { up = 0.8 }
            if del > 0.8 { del = 0.8 }
            if up+del > 0.9 { s := 0.9 / (up + del); up *= s; del *= s }
            insFrac = target - up - del
            if insFrac < 0 { insFrac = 0 }
        } else {
            up = target * 0.20
            del = target * 0.10
            if up > 0.8 { up = 0.8 }
            if del > 0.8 { del = 0.8 }
            if up+del > 0.9 { s := 0.9 / (up + del); up *= s; del *= s }
            insFrac = target - up - del
            if insFrac < 0 { insFrac = 0 }
        }

        mc.Incremental["update_ratio"] = up
        mc.Incremental["delete_ratio"] = del
        mc.Incremental["insert_rows"] = roundInt(rows * insFrac)
        out.Models = append(out.Models, mc)
    }
    return out
}

func findRepoRoot(start string) string {
    dir := start
    for {
        if dir == "/" || dir == "." || dir == "" { break }
        if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
            return dir
        }
        next := filepath.Dir(dir)
        if next == dir { break }
        dir = next
    }
    // fallback to start
    return start
}

func buildIfNeeded(repoRoot string) error {
    cmd := exec.Command("go", "build", "-o", "det-tool")
    cmd.Stdout = os.Stdout
    cmd.Stderr = os.Stderr
    cmd.Dir = repoRoot
    return cmd.Run()
}

func runScenario(repoRoot, cfgPath string, dbCfg string, args []string, logPath string) error {
    if err := os.MkdirAll(filepath.Dir(logPath), 0755); err != nil { return err }
    f, _ := os.Create(logPath)
    defer f.Close()
    cmdArgs := append([]string{"--all", "--config", cfgPath, "--db-config", dbCfg}, args...)
    cmd := exec.Command("./det-tool", cmdArgs...)
    cmd.Stdout = f
    cmd.Stderr = f
    cmd.Env = os.Environ()
    cmd.Dir = repoRoot
    return cmd.Run()
}

func newestReports(dir string, ext string, since time.Time) []fs.FileInfo {
    entries, _ := os.ReadDir(dir)
    files := make([]fs.FileInfo, 0)
    for _, e := range entries {
        if e.IsDir() { continue }
        name := e.Name()
        if strings.HasPrefix(name, "report_execution_") && strings.HasSuffix(name, ext) {
            info, _ := e.Info()
            if info.ModTime().After(since.Add(-10 * time.Second)) {
                files = append(files, info)
            }
        }
    }
    sort.Slice(files, func(i, j int) bool { return files[i].ModTime().After(files[j].ModTime()) })
    return files
}

func copyFile(src, dst string) error {
    in, err := os.ReadFile(src)
    if err != nil { return err }
    if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil { return err }
    return os.WriteFile(dst, in, 0644)
}

// Minimal HTML escaping
func escapeHTML(s string) string {
    r := strings.ReplaceAll(s, "&", "&amp;")
    r = strings.ReplaceAll(r, "<", "&lt;")
    r = strings.ReplaceAll(r, ">", "&gt;")
    r = strings.ReplaceAll(r, "\"", "&quot;")
    r = strings.ReplaceAll(r, "'", "&#39;")
    return r
}

// writeHTMLSummary writes a simple HTML page including:
// - links to each scenario's HTML report (report_XX.html)
// - a Simplified table across ratios
// - a Pivot-by-SQL table with metrics columns per ratio
func writeHTMLSummary(outPath string, labelsOrder []string, allRows []QueryRow) error {
    // Build links to per-scenario HTML reports
    var linksSB strings.Builder
    linksSB.WriteString("<ul>\n")
    for _, lab := range labelsOrder {
        name := fmt.Sprintf("report_%s.html", lab)
        path := filepath.Join(outPath, name)
        if _, err := os.Stat(path); err == nil {
            linksSB.WriteString(fmt.Sprintf("  <li><a href=\"%s\">Scenario %s Report</a></li>\n", name, lab))
        } else {
            linksSB.WriteString(fmt.Sprintf("  <li>Scenario %s Report (not found)</li>\n", lab))
        }
    }
    linksSB.WriteString("</ul>\n")

    // Build Simplified table (by rows)
    var simpleSB strings.Builder
    simpleSB.WriteString("<table>\n<thead><tr>")
    headers := []string{"Ratio", "Model", "Stats Healthy", "Modify Ratio", "Query Label", "Est Error Ratio", "Est Error Value", "Query SQL"}
    for _, h := range headers { simpleSB.WriteString("<th>" + h + "</th>") }
    simpleSB.WriteString("</tr></thead>\n<tbody>\n")
    // Sort rows: by Ratio, Model, Query Label
    rows := make([]QueryRow, len(allRows))
    copy(rows, allRows)
    sort.Slice(rows, func(i, j int) bool {
        if rows[i].Label != rows[j].Label { return rows[i].Label < rows[j].Label }
        if rows[i].Model != rows[j].Model { return rows[i].Model < rows[j].Model }
        if rows[i].QueryLabel != rows[j].QueryLabel { return rows[i].QueryLabel < rows[j].QueryLabel }
        return rows[i].QuerySQL < rows[j].QuerySQL
    })
    for _, r := range rows {
        simpleSB.WriteString("<tr>")
        simpleSB.WriteString(fmt.Sprintf("<td>%s</td>", escapeHTML(r.Label)))
        simpleSB.WriteString(fmt.Sprintf("<td>%s</td>", escapeHTML(r.Model)))
        simpleSB.WriteString(fmt.Sprintf("<td>%d</td>", r.StatsHealthy))
        simpleSB.WriteString(fmt.Sprintf("<td>%.3f</td>", r.ModifyRatio))
        simpleSB.WriteString(fmt.Sprintf("<td>%s</td>", escapeHTML(r.QueryLabel)))
        simpleSB.WriteString(fmt.Sprintf("<td>%.2f</td>", r.EstErrRatio))
        simpleSB.WriteString(fmt.Sprintf("<td>%.2f</td>", r.EstErrValue))
        simpleSB.WriteString(fmt.Sprintf("<td class=\"sql\">%s</td>", escapeHTML(r.QuerySQL)))
        simpleSB.WriteString("</tr>\n")
    }
    simpleSB.WriteString("</tbody>\n</table>\n")

    // Build Pivot table grouped by Query Label (instead of raw SQL)
    type metrics struct{ stats int; mod float64; ratio float64; value float64; bad bool }
    type pivotRow struct{ model, qlabel, sql string; by map[string]metrics; anyBad bool; maxBadRatio float64 }
    pivot := make(map[string]*pivotRow)
    for _, r := range allRows {
        // Group by Model + Query Label to avoid cross-model collisions
        key := r.Model + "|" + r.QueryLabel
        pr, ok := pivot[key]
        if !ok {
            pr = &pivotRow{model: r.Model, qlabel: r.QueryLabel, sql: r.QuerySQL, by: make(map[string]metrics)}
            pivot[key] = pr
        }
        // If multiple SQLs share same label, keep the first seen as a sample
        if pr.sql == "" { pr.sql = r.QuerySQL }
        // bad case: EstErrRatio >= 10 && EstErrValue >= 1000
        m := metrics{r.StatsHealthy, r.ModifyRatio, r.EstErrRatio, r.EstErrValue, (r.EstErrRatio >= 10 && r.EstErrValue >= 1000)}
        pr.by[r.Label] = m
        if m.bad {
            pr.anyBad = true
            if m.ratio > pr.maxBadRatio { pr.maxBadRatio = m.ratio }
        }
    }
    // Color palette for ratio columns (cycled when labels exceed palette)
    colors := []string{"#FFF3E0", "#E3F2FD", "#F1F8E9", "#FCE4EC", "#EDE7F6", "#E0F7FA", "#E8EAF6", "#F3E5F5"}

    // Emit HTML table
    var pivotSB strings.Builder
    pivotSB.WriteString("<table>\n<thead><tr>")
    pheaders := []string{"Model", "Query Label"}
    for _, h := range pheaders { pivotSB.WriteString("<th>" + h + "</th>") }
    for _, lab := range labelsOrder {
        pivotSB.WriteString("<th>StatsHealthy_" + lab + "</th>")
        pivotSB.WriteString("<th>ModifyRatio_" + lab + "</th>")
        pivotSB.WriteString("<th>EstErrRatio_" + lab + "</th>")
        pivotSB.WriteString("<th>EstErrValue_" + lab + "</th>")
    }
    pivotSB.WriteString("<th>Query SQL</th>")
    pivotSB.WriteString("</tr></thead>\n<tbody>\n")
    keys := make([]string, 0, len(pivot))
    for k := range pivot { keys = append(keys, k) }
    sort.Slice(keys, func(i, j int) bool {
        pi, pj := pivot[keys[i]], pivot[keys[j]]
        if pi.anyBad != pj.anyBad { return pi.anyBad && !pj.anyBad }
        if pi.anyBad && pj.anyBad {
            if pi.maxBadRatio != pj.maxBadRatio { return pi.maxBadRatio > pj.maxBadRatio }
        }
        if pi.model != pj.model { return pi.model < pj.model }
        return pi.qlabel < pj.qlabel
    })
    for _, k := range keys {
        pr := pivot[k]
        pivotSB.WriteString("<tr>")
        pivotSB.WriteString("<td>" + escapeHTML(pr.model) + "</td>")
        pivotSB.WriteString("<td>" + escapeHTML(pr.qlabel) + "</td>")
        for i, lab := range labelsOrder {
            bg := colors[i%len(colors)]
            m, ok := pr.by[lab]
            if ok {
                txt := ""
                if m.bad { txt = "color:#d32f2f;font-weight:600;" }
                pivotSB.WriteString(fmt.Sprintf("<td style=\"background:%s;%s\">%d</td>", bg, txt, m.stats))
                pivotSB.WriteString(fmt.Sprintf("<td style=\"background:%s;%s\">%.3f</td>", bg, txt, m.mod))
                pivotSB.WriteString(fmt.Sprintf("<td style=\"background:%s;%s\">%.2f</td>", bg, txt, m.ratio))
                pivotSB.WriteString(fmt.Sprintf("<td style=\"background:%s;%s\">%.2f</td>", bg, txt, m.value))
            } else {
                pivotSB.WriteString(fmt.Sprintf("<td style=\"background:%s;\"></td>", bg))
                pivotSB.WriteString(fmt.Sprintf("<td style=\"background:%s;\"></td>", bg))
                pivotSB.WriteString(fmt.Sprintf("<td style=\"background:%s;\"></td>", bg))
                pivotSB.WriteString(fmt.Sprintf("<td style=\"background:%s;\"></td>", bg))
            }
        }
        pivotSB.WriteString("<td class=\"sql\">" + escapeHTML(pr.sql) + "</td>")
        pivotSB.WriteString("</tr>\n")
    }
    pivotSB.WriteString("</tbody>\n</table>\n")

    // Compose full HTML
    html := `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Scenario Summary</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Microsoft YaHei', sans-serif; margin: 20px; background: #f7f7f7; }
    .container { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
    h1 { margin-top: 0; }
    h2 { margin-top: 28px; }
    ul { line-height: 1.8; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }
    th, td { border: 1px solid #e5e5e5; padding: 6px 8px; text-align: left; vertical-align: top; }
    th { background: #4CAF50; color: #fff; position: sticky; top: 0; }
    tr:nth-child(even) { background: #fafafa; }
    .sql { font-family: Consolas, Menlo, Monaco, 'Courier New', monospace; white-space: pre-wrap; word-break: normal; overflow-wrap: anywhere; min-width: 40ch; }
  </style>
  </head>
<body>
  <div class="container">
    <h1>Scenario Summary</h1>
    %PIVOT_TABLE%
  </div>
</body>
</html>`

    html = strings.ReplaceAll(html, "%PIVOT_TABLE%", pivotSB.String())
    outFile := filepath.Join(outPath, "summary.html")
    return os.WriteFile(outFile, []byte(html), 0644)
}

// Minimal JSON structures for parsing det-tool report JSON
type reportJSON struct {
    TotalQueries int `json:"total_queries"`
    Results []struct {
        Model        string  `json:"model"`
        ModifyRatio  float64 `json:"modify_ratio"`
        StatsHealthy int     `json:"stats_healthy"`
        // below fields are present in det-tool JSON report (Extended + QueryResult)
        QueryLabel           string  `json:"query_label"`
        EstimationErrorRatio float64 `json:"estimation_error_ratio"`
        EstimationErrorValue float64 `json:"estimation_error_value"`
        Query                string  `json:"query"`
    } `json:"results"`
    Summary map[string]interface{} `json:"summary"`
}

type ScenarioSummary struct {
    Label        string                                 `json:"label"`
    UseActual    bool                                   `json:"use_actual"`
    TotalQueries int                                    `json:"total_queries"`
    SuccessRate  float64                                `json:"success_rate"`
    BadCases     int                                    `json:"bad_cases"`
    Models       map[string]map[string]interface{}       `json:"models"`
}

func main() {
    baseCfg := flag.String("config", "config.json", "Base config file")
    dbCfg := flag.String("db-config", "db_config.json", "DB config file")
    includeExternal := flag.Bool("include-external", false, "Include external models (tpcc/tpch)")
    ratiosStr := flag.String("ratios", "0.2,0.5,0.8", "Comma-separated target modify ratios")
    insertMode := flag.String("inc-insert-mode", "load", "insert|load for incremental inserts")
    insertBatch := flag.Int("insert-batch-size", 5000, "Rows per multi-row INSERT")
    dmlBatch := flag.Int("dml-batch-size", 2000, "Statements per TX commit when executing SQL file")
    outDir := flag.String("out", "runs", "Output directory to store logs and copied reports")
    toolOutDir := flag.String("tool-output-dir", "output", "det-tool --output-dir value to collect reports from")
    reportUseActual := flag.Bool("report-use-actual-inc", false, "Pass --report-use-actual-inc to det-tool so reports show actual modify ratio")
    flag.Parse()

    cwd, _ := os.Getwd()
    repoRoot := findRepoRoot(cwd)

    // Normalize paths relative to repo root
    cfgPath := *baseCfg
    if !filepath.IsAbs(cfgPath) { cfgPath = filepath.Join(repoRoot, cfgPath) }
    dbPath := *dbCfg
    if !filepath.IsAbs(dbPath) { dbPath = filepath.Join(repoRoot, dbPath) }
    outPath := *outDir
    if !filepath.IsAbs(outPath) { outPath = filepath.Join(repoRoot, outPath) }

    if err := os.MkdirAll(outPath, 0755); err != nil { fmt.Println("mkdir:", err); os.Exit(1) }

    var cfg itypes.Config
    if err := readJSON(cfgPath, &cfg); err != nil {
        fmt.Println("read base config:", err)
        os.Exit(1)
    }

    if err := buildIfNeeded(repoRoot); err != nil {
        fmt.Println("build det-tool:", err)
        os.Exit(1)
    }

    parts := strings.Split(*ratiosStr, ",")
    ratios := make([]float64, 0, len(parts))
    for _, p := range parts {
        var f float64
        fmt.Sscanf(strings.TrimSpace(p), "%f", &f)
        ratios = append(ratios, f)
    }

    summaries := make([]ScenarioSummary, 0, len(ratios))
    // Collect per-query rows across ratios for simplified report
    var allRows []QueryRow
    labelsOrder := make([]string, 0, len(ratios))
    for _, r := range ratios {
        label := fmt.Sprintf("%d", int(r*100+0.5))
        labelsOrder = append(labelsOrder, label)
        scenCfg := adjustConfigForRatio(cfg, r, *includeExternal)
        scenCfgPath := filepath.Join(outPath, fmt.Sprintf("config_scenario_%s.json", label))
        if err := writeJSON(scenCfgPath, scenCfg); err != nil {
            fmt.Println("write config:", err)
            continue
        }
        logPath := filepath.Join(outPath, fmt.Sprintf("scenario_%s.log", label))
        start := time.Now()
        runArgs := []string{"--inc-insert-mode", *insertMode, "--insert-batch-size", fmt.Sprintf("%d", *insertBatch), "--dml-batch-size", fmt.Sprintf("%d", *dmlBatch), "--output-dir", *toolOutDir}
        if *reportUseActual { runArgs = append(runArgs, "--report-use-actual-inc") }
        fmt.Printf("[Runner] Executing scenario %s (ratio=%.2f) with %s\n", label, r, scenCfgPath)
        if err := runScenario(repoRoot, scenCfgPath, dbPath, runArgs, logPath); err != nil {
            fmt.Println("scenario run:", err)
        }

        // Collect reports from det-tool output directory
        reportsDir := filepath.Join(repoRoot, *toolOutDir, "reports")
        newest := newestReports(reportsDir, ".html", start)
        if len(newest) > 0 {
            src := filepath.Join(reportsDir, newest[0].Name())
            dst := filepath.Join(outPath, fmt.Sprintf("report_%s.html", label))
            _ = copyFile(src, dst)
        }
        newestCSV := newestReports(reportsDir, ".csv", start)
        if len(newestCSV) > 0 {
            src := filepath.Join(reportsDir, newestCSV[0].Name())
            dst := filepath.Join(outPath, fmt.Sprintf("report_%s.csv", label))
            _ = copyFile(src, dst)
        }
        newestJSON := newestReports(reportsDir, ".json", start)
        if len(newestJSON) > 0 {
            src := filepath.Join(reportsDir, newestJSON[0].Name())
            dst := filepath.Join(outPath, fmt.Sprintf("report_%s.json", label))
            _ = copyFile(src, dst)
            // Parse and summarize
            var rep reportJSON
            if err := readJSON(dst, &rep); err == nil {
                sum := ScenarioSummary{Label: label, UseActual: *reportUseActual, TotalQueries: rep.TotalQueries}
                if rep.Summary != nil {
                    if v, ok := rep.Summary["success_rate"].(float64); ok { sum.SuccessRate = v }
                    if v, ok := rep.Summary["bad_cases"].(float64); ok { sum.BadCases = int(v) }
                }
                sum.Models = make(map[string]map[string]interface{})
                for _, r := range rep.Results {
                    if _, ok := sum.Models[r.Model]; !ok {
                        sum.Models[r.Model] = map[string]interface{}{
                            "modify_ratio":  r.ModifyRatio,
                            "stats_healthy": r.StatsHealthy,
                        }
                    }
                    // Append per-query row for simplified report
                    allRows = append(allRows, QueryRow{
                        Label:        label,
                        Model:        r.Model,
                        StatsHealthy: r.StatsHealthy,
                        ModifyRatio:  r.ModifyRatio,
                        QueryLabel:   r.QueryLabel,
                        EstErrRatio:  r.EstimationErrorRatio,
                        EstErrValue:  r.EstimationErrorValue,
                        QuerySQL:     r.Query,
                    })
                }
                summaries = append(summaries, sum)
            }
        }
    }
    if len(summaries) > 0 {
        _ = writeJSON(filepath.Join(outPath, "summary.json"), summaries)
        fmt.Printf("[Runner] Summary written to %s\n", filepath.Join(outPath, "summary.json"))
    }

    // Write simplified per-query report across ratios
    if len(allRows) > 0 {
        simpleCSV := filepath.Join(outPath, "queries_by_ratio.csv")
        f, err := os.Create(simpleCSV)
        if err == nil {
            w := csv.NewWriter(f)
            _ = w.Write([]string{"Ratio", "Model", "Stats Healthy", "Modify Ratio", "Query Label", "Est Error Ratio", "Est Error Value", "Query SQL"})
            for _, row := range allRows {
                _ = w.Write([]string{
                    row.Label,
                    row.Model,
                    fmt.Sprintf("%d", row.StatsHealthy),
                    fmt.Sprintf("%.3f", row.ModifyRatio),
                    row.QueryLabel,
                    fmt.Sprintf("%.2f", row.EstErrRatio),
                    fmt.Sprintf("%.2f", row.EstErrValue),
                    row.QuerySQL,
                })
            }
            w.Flush()
            f.Close()
            fmt.Printf("[Runner] Wrote simplified report: %s\n", simpleCSV)
        }

        // Pivot by Query SQL (optionally include label-specific metrics)
        type metrics struct{ stats int; mod float64; ratio float64; value float64 }
        pivot := make(map[string]struct{
            label string
            model string
            qlabel string
            by map[string]metrics
        })
        for _, row := range allRows {
            key := row.QuerySQL
            p, ok := pivot[key]
            if !ok { p = struct{ label string; model string; qlabel string; by map[string]metrics }{"", row.Model, row.QueryLabel, make(map[string]metrics)} }
            p.by[row.Label] = metrics{row.StatsHealthy, row.ModifyRatio, row.EstErrRatio, row.EstErrValue}
            pivot[key] = p
        }
        pivotCSV := filepath.Join(outPath, "queries_pivot.csv")
        pf, err := os.Create(pivotCSV)
        if err == nil {
            w := csv.NewWriter(pf)
            // Header: Query SQL, Query Label, Model, then groups per label
            header := []string{"Query SQL", "Query Label", "Model"}
            for _, lab := range labelsOrder {
                header = append(header, "StatsHealthy_"+lab, "ModifyRatio_"+lab, "EstErrRatio_"+lab, "EstErrValue_"+lab)
            }
            _ = w.Write(header)
            // Rows
            keys := make([]string, 0, len(pivot))
            for k := range pivot { keys = append(keys, k) }
            sort.Strings(keys)
            for _, k := range keys {
                p := pivot[k]
                rec := []string{k, p.qlabel, p.model}
                for _, lab := range labelsOrder {
                    m, ok := p.by[lab]
                    if ok {
                        rec = append(rec, fmt.Sprintf("%d", m.stats), fmt.Sprintf("%.3f", m.mod), fmt.Sprintf("%.2f", m.ratio), fmt.Sprintf("%.2f", m.value))
                    } else {
                        rec = append(rec, "", "", "", "")
                    }
                }
                _ = w.Write(rec)
            }
            w.Flush()
            pf.Close()
            fmt.Printf("[Runner] Wrote pivot report: %s\n", pivotCSV)
        }

        // Write HTML summary including links + simplified + pivot
        if err := writeHTMLSummary(outPath, labelsOrder, allRows); err == nil {
            fmt.Printf("[Runner] Wrote HTML summary: %s\n", filepath.Join(outPath, "summary.html"))
        } else {
            fmt.Println("[Runner] Failed to write HTML summary:", err)
        }
    }
}
