package main

import (
	"database/sql"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode"

	itypes "det-tool/internal/types"
	_ "github.com/go-sql-driver/mysql"
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

type IssueCase struct {
	Label            string
	Model            string
	QueryLabel       string
	QuerySQL         string
	EstErrRatio      float64
	EstErrValue      float64
	Explain          string
	PlanReplayerLink string
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
	if m == nil {
		return 0
	}
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

func roundInt(x float64) int {
	if x < 0 {
		return 0
	}
	return int(x + 0.5)
}

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
		for k, v := range m.Params {
			mc.Params[k] = v
		}
		for k, v := range m.Incremental {
			mc.Incremental[k] = v
		}

		rows := getFloat(mc.Params, "rows")
		if rows <= 0 {
			rows = 1000
		}

		baseInsert := getFloat(mc.Incremental, "insert_rows") / rows
		baseUpdate := getFloat(mc.Incremental, "update_ratio")
		baseDelete := getFloat(mc.Incremental, "delete_ratio")
		baseSum := baseInsert + baseUpdate + baseDelete

		var up, del, insFrac float64
		if baseSum > 0 {
			scale := target / baseSum
			up = baseUpdate * scale
			del = baseDelete * scale
			if up > 0.8 {
				up = 0.8
			}
			if del > 0.8 {
				del = 0.8
			}
			if up+del > 0.9 {
				s := 0.9 / (up + del)
				up *= s
				del *= s
			}
			insFrac = target - up - del
			if insFrac < 0 {
				insFrac = 0
			}
		} else {
			up = target * 0.20
			del = target * 0.10
			if up > 0.8 {
				up = 0.8
			}
			if del > 0.8 {
				del = 0.8
			}
			if up+del > 0.9 {
				s := 0.9 / (up + del)
				up *= s
				del *= s
			}
			insFrac = target - up - del
			if insFrac < 0 {
				insFrac = 0
			}
		}

		mc.Incremental["update_ratio"] = up
		mc.Incremental["delete_ratio"] = del
		mc.Incremental["insert_rows"] = roundInt(rows * insFrac)

		// If external TPCC is included, set its run time in gen-inc to ratio*10 minutes
		if strings.HasPrefix(mc.Type, "external_tpcc") {
			mins := int(math.Round(target * 10.0))
			if mins < 1 {
				mins = 1
			}
			mc.Incremental["time"] = fmt.Sprintf("%dm", mins)
		}
		out.Models = append(out.Models, mc)
	}
	return out
}

func findRepoRoot(start string) string {
	dir := start
	for {
		if dir == "/" || dir == "." || dir == "" {
			break
		}
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		next := filepath.Dir(dir)
		if next == dir {
			break
		}
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
	if err := os.MkdirAll(filepath.Dir(logPath), 0755); err != nil {
		return err
	}
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
		if e.IsDir() {
			continue
		}
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
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
		return err
	}
	return os.WriteFile(dst, in, 0644)
}

func sanitizeFileName(s string) string {
	if s == "" {
		return "issue"
	}
	s = strings.ToLower(s)
	var b strings.Builder
	lastUnderscore := false
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
			lastUnderscore = false
			continue
		}
		if !lastUnderscore {
			b.WriteByte('_')
			lastUnderscore = true
		}
	}
	out := strings.Trim(b.String(), "_")
	if out == "" {
		return "issue"
	}
	return out
}

func loadExistingIssueKeys(dir string) map[string]bool {
	keys := map[string]bool{}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return keys
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") {
			continue
		}
		path := filepath.Join(dir, e.Name())
		b, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var key string
		var title string
		for _, line := range strings.Split(string(b), "\n") {
			if strings.HasPrefix(line, "Key:") {
				key = strings.TrimSpace(strings.TrimPrefix(line, "Key:"))
			}
			if strings.HasPrefix(line, "Title:") {
				title = strings.TrimSpace(strings.TrimPrefix(line, "Title:"))
			}
			if key != "" {
				break
			}
		}
		if key == "" && title != "" {
			if idx := strings.LastIndex(title, "-"); idx > 0 {
				key = title[:idx]
			}
		}
		if key != "" {
			keys[key] = true
		}
	}
	return keys
}

func extractStatsMeta(logPath string) string {
	b, err := os.ReadFile(logPath)
	if err != nil {
		return ""
	}
	lines := strings.Split(string(b), "\n")
	start := -1
	for i, line := range lines {
		if strings.HasPrefix(line, "[DB] STATS_META for DB=") {
			start = i
		}
	}
	if start == -1 {
		return ""
	}
	end := len(lines)
	for i := start + 1; i < len(lines); i++ {
		trimmed := strings.TrimSpace(lines[i])
		if trimmed == "" || strings.HasPrefix(trimmed, "===") {
			end = i
			break
		}
	}
	return strings.Join(lines[start:end], "\n")
}

func fetchTiDBVersion(cfg itypes.DBConfig) string {
	dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/?charset=%s&parseTime=true&loc=Local",
		cfg.User, cfg.Password, cfg.Host, cfg.Port, cfg.Charset)
	db, err := sql.Open("mysql", dsn)
	if err != nil {
		return ""
	}
	defer db.Close()
	var v sql.NullString
	if err := db.QueryRow("select tidb_version()").Scan(&v); err != nil {
		return ""
	}
	if v.Valid {
		return v.String
	}
	return ""
}

func generateIssueTemplates(outPath string, cases []IssueCase, statsMetaByLabel map[string]string, tidbVersion string) {
	if len(cases) == 0 {
		return
	}
	issuesDir := filepath.Join(outPath, "issues")
	if err := os.MkdirAll(issuesDir, 0755); err != nil {
		fmt.Println("[Runner] Failed to create issues dir:", err)
		return
	}
	seen := loadExistingIssueKeys(issuesDir)
	for _, c := range cases {
		key := c.Model + "|" + c.QueryLabel
		if seen[key] {
			continue
		}
		seen[key] = true
		title := fmt.Sprintf("%s-%s-%s%%", c.Model, c.QueryLabel, c.Label)
		plan := c.PlanReplayerLink
		if plan == "" {
			plan = "../planreplayer/replayer_xxx.zip"
		}
		statsMeta := statsMetaByLabel[c.Label]
		if statsMeta == "" {
			statsMeta = "show stats_meta; -- not available"
		}
		version := tidbVersion
		if version == "" {
			version = "select tidb_version(); -- not available"
		}
		explain := c.Explain
		if explain == "" {
			explain = "N/A"
		}
		body := strings.Join([]string{
			"Title: " + title,
			"Key: " + key,
			"",
			"### 1. Minimal reproduce step (Required)",
			"```sql",
			"drop database if exists det_test_db;",
			"```",
			"",
			"```",
			"plan replayer load '" + plan + "' -- 本case 对应生成的planreplayer 文件",
			"```",
			"",
			"```sql",
			c.QuerySQL,
			"```",
			"",
			"```txt",
			statsMeta,
			"```",
			"",
			"### 2. What did you expect to see? (Required)",
			"Est Error Ratio<10 and Est Error Value<1000",
			"",
			"### 3. What did you see instead (Required)",
			fmt.Sprintf("Est Error Ratio: %.2f", c.EstErrRatio),
			fmt.Sprintf("Est Error Value: %.2f", c.EstErrValue),
			"Explain Plan:",
			"```txt",
			explain,
			"```",
			"",
			"### 4. What is your TiDB version? (Required)",
			"```txt",
			version,
			"```",
			"",
		}, "\n")
		fileName := fmt.Sprintf("issue_%s.md", sanitizeFileName(c.Model+"_"+c.QueryLabel))
		dst := filepath.Join(issuesDir, fileName)
		if err := os.WriteFile(dst, []byte(body), 0644); err != nil {
			fmt.Println("[Runner] Failed to write issue template:", err)
			continue
		}
		fmt.Printf("[Runner] Wrote issue template: %s\n", dst)
	}
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
	for _, h := range headers {
		simpleSB.WriteString("<th>" + h + "</th>")
	}
	simpleSB.WriteString("</tr></thead>\n<tbody>\n")
	// Sort rows: by Ratio, Model, Query Label
	rows := make([]QueryRow, len(allRows))
	copy(rows, allRows)
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Label != rows[j].Label {
			return rows[i].Label < rows[j].Label
		}
		if rows[i].Model != rows[j].Model {
			return rows[i].Model < rows[j].Model
		}
		if rows[i].QueryLabel != rows[j].QueryLabel {
			return rows[i].QueryLabel < rows[j].QueryLabel
		}
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
	type metrics struct {
		stats int
		mod   float64
		ratio float64
		value float64
		bad   bool
	}
	type pivotRow struct {
		model       string
		qlabel      string
		sql         string
		by          map[string]metrics
		anyBad      bool
		maxBadRatio float64
		maxErrRatio float64
		maxErrValue float64
		maxErrLabel string
	}
	pivot := make(map[string]*pivotRow)
	for _, r := range allRows {
		// Group by Model + Query Label; if Query Label is empty (e.g., external tpcc),
		// fall back to using the SQL text for grouping to avoid collapsing multiple queries.
		glabel := r.QueryLabel
		if strings.TrimSpace(glabel) == "" {
			glabel = r.QuerySQL
		}
		key := r.Model + "|" + glabel
		pr, ok := pivot[key]
		if !ok {
			pr = &pivotRow{model: r.Model, qlabel: glabel, sql: r.QuerySQL, by: make(map[string]metrics)}
			pivot[key] = pr
		}
		// If multiple SQLs share same label, keep the first seen as a sample
		if pr.sql == "" {
			pr.sql = r.QuerySQL
		}
		// bad case: EstErrRatio >= 10 && EstErrValue >= 1000
		m := metrics{r.StatsHealthy, r.ModifyRatio, r.EstErrRatio, r.EstErrValue, (r.EstErrRatio >= 10 && r.EstErrValue >= 1000)}
		pr.by[r.Label] = m
		if m.bad {
			pr.anyBad = true
			if m.ratio > pr.maxBadRatio {
				pr.maxBadRatio = m.ratio
			}
		}
		if m.ratio > pr.maxErrRatio {
			pr.maxErrRatio = m.ratio
			pr.maxErrValue = m.value
			pr.maxErrLabel = r.Label
		}
	}
	// Color palette for ratio columns (cycled when labels exceed palette)
	colors := []string{"#FFF3E0", "#E3F2FD", "#F1F8E9", "#FCE4EC", "#EDE7F6", "#E0F7FA", "#E8EAF6", "#F3E5F5"}

	// Emit HTML table
	var pivotSB strings.Builder
	pivotSB.WriteString("<table>\n<thead><tr>")
	pheaders := []string{"Model", "Query Label", "Bug?", "Issue"}
	for _, h := range pheaders {
		pivotSB.WriteString("<th>" + h + "</th>")
	}
	for _, lab := range labelsOrder {
		pivotSB.WriteString("<th>StatsHealthy_" + lab + "</th>")
		pivotSB.WriteString("<th>ModifyRatio_" + lab + "</th>")
		pivotSB.WriteString("<th>EstErrRatio_" + lab + "</th>")
		pivotSB.WriteString("<th>EstErrValue_" + lab + "</th>")
	}
	pivotSB.WriteString("<th>Query SQL</th>")
	pivotSB.WriteString("</tr></thead>\n<tbody>\n")
	keys := make([]string, 0, len(pivot))
	for k := range pivot {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		pi, pj := pivot[keys[i]], pivot[keys[j]]
		if pi.anyBad != pj.anyBad {
			return pi.anyBad && !pj.anyBad
		}
		if pi.anyBad && pj.anyBad {
			if pi.maxBadRatio != pj.maxBadRatio {
				return pi.maxBadRatio > pj.maxBadRatio
			}
		}
		if pi.model != pj.model {
			return pi.model < pj.model
		}
		return pi.qlabel < pj.qlabel
	})
	for _, k := range keys {
		pr := pivot[k]
		pivotSB.WriteString("<tr>")
		pivotSB.WriteString("<td>" + escapeHTML(pr.model) + "</td>")
		pivotSB.WriteString("<td>" + escapeHTML(pr.qlabel) + "</td>")
		pivotSB.WriteString("<td><select class=\"bug-select\" onchange=\"onBugSelect(this)\">" +
			"<option value=\"\">Unconfirmed</option>" +
			"<option value=\"bug\">Bug</option>" +
			"<option value=\"no\">Not Bug</option>" +
			"</select></td>")
		pivotSB.WriteString(fmt.Sprintf("<td><button class=\"issue-btn\" onclick=\"createIssue(this)\" disabled "+
			"data-model=\"%s\" data-qlabel=\"%s\" data-sql=\"%s\" data-err-ratio=\"%.2f\" data-err-value=\"%.2f\" data-ratio-label=\"%s\">Create Issue</button></td>",
			escapeHTML(pr.model),
			escapeHTML(pr.qlabel),
			escapeHTML(pr.sql),
			pr.maxErrRatio,
			pr.maxErrValue,
			escapeHTML(pr.maxErrLabel),
		))
		for i, lab := range labelsOrder {
			bg := colors[i%len(colors)]
			m, ok := pr.by[lab]
			if ok {
				txt := ""
				if m.bad {
					txt = "color:#d32f2f;font-weight:600;"
				}
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
    .actions { margin: 16px 0 10px; padding: 12px; border: 1px solid #e5e5e5; border-radius: 6px; background: #f8fbff; }
    .actions label { margin-right: 12px; font-size: 12px; }
    .actions input { margin-left: 6px; padding: 4px 6px; font-size: 12px; }
    .actions small { display: block; margin-top: 6px; color: #666; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }
    th, td { border: 1px solid #e5e5e5; padding: 6px 8px; text-align: left; vertical-align: top; }
    th { background: #4CAF50; color: #fff; position: sticky; top: 0; }
    tr:nth-child(even) { background: #fafafa; }
    tr.is-bug { background: #fff4e5; }
    .sql { font-family: Consolas, Menlo, Monaco, 'Courier New', monospace; white-space: pre-wrap; word-break: normal; overflow-wrap: anywhere; min-width: 40ch; }
    .bug-select { font-size: 12px; }
    .issue-btn { font-size: 12px; padding: 4px 8px; background: #1976d2; color: #fff; border: none; border-radius: 4px; cursor: pointer; }
    .issue-btn:disabled { background: #9e9e9e; cursor: not-allowed; }
    .issue-created { background: #2e7d32; }
  </style>
  </head>
<body>
  <div class="container">
    <h1>Scenario Summary</h1>
    <div class="actions">
      <label>GitHub Repo<input id="gh-repo" class="gh-setting" placeholder="owner/repo" /></label>
      <label>Token<input id="gh-token" class="gh-setting" type="password" placeholder="optional" /></label>
      <label>Labels<input id="gh-labels" class="gh-setting" placeholder="bug,det-tool" /></label>
      <small>If token is set, issues are created via GitHub API; otherwise the button opens a prefilled issue page.</small>
    </div>
    %LINKS%
    %PIVOT_TABLE%
  </div>
  <script>
    function byId(id) { return document.getElementById(id); }
    function saveSettings() {
      try {
        var repo = byId('gh-repo').value.trim();
        var token = byId('gh-token').value.trim();
        var labels = byId('gh-labels').value.trim();
        if (repo) { localStorage.setItem('dettool_gh_repo', repo); } else { localStorage.removeItem('dettool_gh_repo'); }
        if (token) { localStorage.setItem('dettool_gh_token', token); } else { localStorage.removeItem('dettool_gh_token'); }
        if (labels) { localStorage.setItem('dettool_gh_labels', labels); } else { localStorage.removeItem('dettool_gh_labels'); }
      } catch (e) {}
    }
    function loadSettings() {
      try {
        var repo = localStorage.getItem('dettool_gh_repo') || '';
        var token = localStorage.getItem('dettool_gh_token') || '';
        var labels = localStorage.getItem('dettool_gh_labels') || '';
        byId('gh-repo').value = repo;
        byId('gh-token').value = token;
        byId('gh-labels').value = labels;
      } catch (e) {}
    }
    function onBugSelect(sel) {
      var tr = sel.closest('tr');
      if (!tr) { return; }
      var btn = tr.querySelector('.issue-btn');
      if (btn) { btn.disabled = (sel.value !== 'bug'); }
      if (sel.value === 'bug') { tr.classList.add('is-bug'); } else { tr.classList.remove('is-bug'); }
    }
    function parseLabels(s) {
      if (!s) { return []; }
      var parts = s.split(',');
      var out = [];
      for (var i = 0; i < parts.length; i++) {
        var v = parts[i].trim();
        if (v) { out.push(v); }
      }
      return out;
    }
    function buildIssueData(btn) {
      var model = btn.getAttribute('data-model') || '';
      var qlabel = btn.getAttribute('data-qlabel') || '';
      var sql = btn.getAttribute('data-sql') || '';
      var ratio = btn.getAttribute('data-err-ratio') || '';
      var value = btn.getAttribute('data-err-value') || '';
      var label = btn.getAttribute('data-ratio-label') || '';
      var titleLabel = qlabel;
      if (titleLabel.length > 80) { titleLabel = titleLabel.slice(0, 77) + '...'; }
      var title = 'det-tool issue';
      if (model && qlabel) {
        title = model + '-' + titleLabel;
      } else if (model) {
        title = model;
      } else if (qlabel) {
        title = titleLabel;
      }
      if (label) { title += '-' + label + '%'; }
      var ratioLabel = label ? (label + '%') : 'n/a';
      var fence = String.fromCharCode(96) + String.fromCharCode(96) + String.fromCharCode(96);
      var fenceSql = fence + 'sql';
      var body = [
        '### Summary',
        '- Model: ' + model,
        '- Query Label: ' + qlabel,
        '- Worst Ratio Label: ' + ratioLabel,
        '- Est Error Ratio: ' + ratio,
        '- Est Error Value: ' + value,
        '',
        '### SQL',
        fenceSql,
        sql,
        fence,
        '',
        '### Notes',
        '- Marked as bug in summary.html'
      ].join('\n');
      return { title: title, body: body };
    }
    function openPrefilledIssue(repo, title, body, labels) {
      var url = 'https://github.com/' + repo + '/issues/new?title=' + encodeURIComponent(title) + '&body=' + encodeURIComponent(body);
      if (labels) { url += '&labels=' + encodeURIComponent(labels); }
      window.open(url, '_blank');
    }
    function createIssue(btn) {
      var repo = byId('gh-repo').value.trim();
      if (!repo) { alert('Set GitHub repo (owner/repo)'); return; }
      var token = byId('gh-token').value.trim();
      var labels = byId('gh-labels').value.trim();
      var data = buildIssueData(btn);
      if (!token) {
        openPrefilledIssue(repo, data.title, data.body, labels);
        return;
      }
      var payload = { title: data.title, body: data.body };
      var labs = parseLabels(labels);
      if (labs.length > 0) { payload.labels = labs; }
      var oldText = btn.textContent;
      btn.disabled = true;
      btn.textContent = 'Creating...';
      fetch('https://api.github.com/repos/' + repo + '/issues', {
        method: 'POST',
        headers: {
          'Accept': 'application/vnd.github+json',
          'Authorization': 'Bearer ' + token
        },
        body: JSON.stringify(payload)
      }).then(function(r) {
        return r.json().then(function(data) { return { ok: r.ok, data: data }; });
      }).then(function(res) {
        if (!res.ok) {
          alert('Create issue failed: ' + (res.data && res.data.message ? res.data.message : 'unknown error'));
          btn.disabled = false;
          btn.textContent = oldText;
          return;
        }
        btn.textContent = 'Created';
        btn.classList.add('issue-created');
        if (res.data && res.data.html_url) {
          window.open(res.data.html_url, '_blank');
        }
      }).catch(function(err) {
        alert('Create issue failed: ' + err);
        btn.disabled = false;
        btn.textContent = oldText;
      });
    }
    document.addEventListener('DOMContentLoaded', function() {
      loadSettings();
      var inputs = document.querySelectorAll('.gh-setting');
      for (var i = 0; i < inputs.length; i++) {
        inputs[i].addEventListener('input', saveSettings);
      }
    });
  </script>
</body>
</html>`

	html = strings.ReplaceAll(html, "%LINKS%", linksSB.String())
	html = strings.ReplaceAll(html, "%PIVOT_TABLE%", pivotSB.String())
	outFile := filepath.Join(outPath, "summary.html")
	return os.WriteFile(outFile, []byte(html), 0644)
}

// Minimal JSON structures for parsing det-tool report JSON
type reportJSON struct {
	TotalQueries int `json:"total_queries"`
	Results      []struct {
		Model        string  `json:"model"`
		ModifyRatio  float64 `json:"modify_ratio"`
		StatsHealthy int     `json:"stats_healthy"`
		IsBadCase    bool    `json:"is_bad_case"`
		// below fields are present in det-tool JSON report (Extended + QueryResult)
		QueryLabel           string  `json:"query_label"`
		EstimationErrorRatio float64 `json:"estimation_error_ratio"`
		EstimationErrorValue float64 `json:"estimation_error_value"`
		Query                string  `json:"query"`
		Explain              string  `json:"explain"`
		PlanReplayerLink     string  `json:"plan_replayer_link"`
	} `json:"results"`
	Summary map[string]interface{} `json:"summary"`
}

type ScenarioSummary struct {
	Label        string                            `json:"label"`
	UseActual    bool                              `json:"use_actual"`
	TotalQueries int                               `json:"total_queries"`
	SuccessRate  float64                           `json:"success_rate"`
	BadCases     int                               `json:"bad_cases"`
	Models       map[string]map[string]interface{} `json:"models"`
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
	if !filepath.IsAbs(cfgPath) {
		cfgPath = filepath.Join(repoRoot, cfgPath)
	}
	dbPath := *dbCfg
	if !filepath.IsAbs(dbPath) {
		dbPath = filepath.Join(repoRoot, dbPath)
	}
	outPath := *outDir
	if !filepath.IsAbs(outPath) {
		outPath = filepath.Join(repoRoot, outPath)
	}

	if err := os.MkdirAll(outPath, 0755); err != nil {
		fmt.Println("mkdir:", err)
		os.Exit(1)
	}

	var cfg itypes.Config
	if err := readJSON(cfgPath, &cfg); err != nil {
		fmt.Println("read base config:", err)
		os.Exit(1)
	}
	var dbCfgObj itypes.DBConfig
	if err := readJSON(dbPath, &dbCfgObj); err != nil {
		fmt.Println("read db config:", err)
	}
	tidbVersion := fetchTiDBVersion(dbCfgObj)

	if err := buildIfNeeded(repoRoot); err != nil {
		fmt.Println("build det-tool:", err)
		os.Exit(1)
	}

	ratios := []float64{}
	if *ratiosStr != "" {
		parts := strings.Split(*ratiosStr, ",")
		ratios = make([]float64, 0, len(parts))
		for _, p := range parts {
			var f float64
			fmt.Sscanf(strings.TrimSpace(p), "%f", &f)
			ratios = append(ratios, f)
		}
	} else {
		// Derive a single ratio from config's average target modify ratio when --ratios is not specified
		avg := 0.0
		// lightweight re-calc here mirroring report's calculateModifyRatio logic
		for _, m := range cfg.Models {
			if strings.HasPrefix(m.Type, "external_") {
				continue
			}
			rows := getFloat(m.Params, "rows")
			if rows <= 0 {
				rows = 1000
			}
			ins := getFloat(m.Incremental, "insert_rows") / rows
			up := getFloat(m.Incremental, "update_ratio")
			del := getFloat(m.Incremental, "delete_ratio")
			avg += ins + up + del
		}
		if len(cfg.Models) > 0 {
			avg = avg / float64(len(cfg.Models))
		}
		if avg <= 0 {
			avg = 0.2
		} // sensible default
		ratios = []float64{avg}
	}

	summaries := make([]ScenarioSummary, 0, len(ratios))
	// Collect per-query rows across ratios for simplified report
	var allRows []QueryRow
	var issueCases []IssueCase
	labelsOrder := make([]string, 0, len(ratios))
	statsMetaByLabel := map[string]string{}
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
		if *reportUseActual {
			runArgs = append(runArgs, "--report-use-actual-inc")
		}
		fmt.Printf("[Runner] Executing scenario %s (ratio=%.2f) with %s\n", label, r, scenCfgPath)
		if err := runScenario(repoRoot, scenCfgPath, dbPath, runArgs, logPath); err != nil {
			fmt.Println("scenario run:", err)
		}
		statsMetaByLabel[label] = extractStatsMeta(logPath)

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
					if v, ok := rep.Summary["success_rate"].(float64); ok {
						sum.SuccessRate = v
					}
					if v, ok := rep.Summary["bad_cases"].(float64); ok {
						sum.BadCases = int(v)
					}
				}
				sum.Models = make(map[string]map[string]interface{})
				for _, r := range rep.Results {
					if _, ok := sum.Models[r.Model]; !ok {
						sum.Models[r.Model] = map[string]interface{}{
							"modify_ratio":  r.ModifyRatio,
							"stats_healthy": r.StatsHealthy,
						}
					}
					isBad := r.IsBadCase || (r.EstimationErrorRatio >= 10 && r.EstimationErrorValue >= 1000)
					if isBad {
						issueCases = append(issueCases, IssueCase{
							Label:            label,
							Model:            r.Model,
							QueryLabel:       r.QueryLabel,
							QuerySQL:         r.Query,
							EstErrRatio:      r.EstimationErrorRatio,
							EstErrValue:      r.EstimationErrorValue,
							Explain:          r.Explain,
							PlanReplayerLink: r.PlanReplayerLink,
						})
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
		type metrics struct {
			stats int
			mod   float64
			ratio float64
			value float64
		}
		pivot := make(map[string]struct {
			label  string
			model  string
			qlabel string
			by     map[string]metrics
		})
		for _, row := range allRows {
			key := row.QuerySQL
			p, ok := pivot[key]
			if !ok {
				p = struct {
					label  string
					model  string
					qlabel string
					by     map[string]metrics
				}{"", row.Model, row.QueryLabel, make(map[string]metrics)}
			}
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
			for k := range pivot {
				keys = append(keys, k)
			}
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

	if len(issueCases) > 0 {
		generateIssueTemplates(outPath, issueCases, statsMetaByLabel, tidbVersion)
	}
}
