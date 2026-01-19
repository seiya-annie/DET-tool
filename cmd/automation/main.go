package main

import (
	"crypto/sha1"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	itypes "det-tool/internal/types"
	_ "github.com/go-sql-driver/mysql"
)

// minimal struct mirroring cmd/scenarios report JSON
type reportJSON struct {
	Results []struct {
		Model                string  `json:"model"`
		ModifyRatio          float64 `json:"modify_ratio"`
		QueryLabel           string  `json:"query_label"`
		EstimationErrorRatio float64 `json:"estimation_error_ratio"`
		EstimationErrorValue float64 `json:"estimation_error_value"`
		Query                string  `json:"query"`
		Explain              string  `json:"explain"`
		PlanReplayerLink     string  `json:"plan_replayer_link"`
	} `json:"results"`
}

// rulesState tracks last processed rules file info
type rulesState struct {
	FileSHA1  string   `json:"file_sha1"`
	Labels    []string `json:"labels"`
	UpdatedAt string   `json:"updated_at"`
}

func fileSHA1(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha1.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
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

func buildBinaries(repoRoot string) error {
	// build det-tool
	c1 := exec.Command("go", "build", "-o", "det-tool")
	c1.Stdout, c1.Stderr, c1.Dir = os.Stdout, os.Stderr, repoRoot
	if err := c1.Run(); err != nil {
		return err
	}
	// build run-scenarios
	c2 := exec.Command("go", "build", "-o", filepath.Join("cmd", "scenarios", "run-scenarios"), "./cmd/scenarios")
	c2.Stdout, c2.Stderr, c2.Dir = os.Stdout, os.Stderr, repoRoot
	return c2.Run()
}

func runScenarios(repoRoot, cfg, dbCfg, toolOut, runsOut, ratios string) error {
	args := []string{
		filepath.Join("cmd", "scenarios", "run-scenarios"),
		"--config", cfg,
		"--db-config", dbCfg,
		"--ratios", ratios,
		"--inc-insert-mode", "load",
		"--insert-batch-size", "2000",
		"--dml-batch-size", "1000",
		"--tool-output-dir", toolOut,
		"--out", runsOut,
		"--report-use-actual-inc",
	}
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Stdout, cmd.Stderr, cmd.Dir = os.Stdout, os.Stderr, repoRoot
	return cmd.Run()
}

func extractTokenFromLink(link string) string {
	// expected like ../planreplayer/<token>.zip or absolute URL
	if link == "" {
		return ""
	}
	if strings.HasPrefix(link, "http://") || strings.HasPrefix(link, "https://") {
		parts := strings.Split(link, "/")
		if len(parts) > 0 {
			return strings.TrimSuffix(parts[len(parts)-1], ".zip")
		}
		return ""
	}
	base := filepath.Base(link)
	return strings.TrimSuffix(base, ".zip")
}

func normalizePlanReplayerPath(link string) string {
	// return repository-relative path when possible
	token := extractTokenFromLink(link)
	if token == "" {
		return link
	}
	return filepath.ToSlash(filepath.Join("output", "planreplayer", token+".zip"))
}

func fetchTiDBVersion(db itypes.DBConfig) string {
	dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/?charset=%s&parseTime=true&loc=Local", db.User, db.Password, db.Host, db.Port, db.Charset)
	sqlDB, err := sql.Open("mysql", dsn)
	if err != nil {
		return fmt.Sprintf("open error: %v", err)
	}
	defer sqlDB.Close()
	var v string
	if err := sqlDB.QueryRow("SELECT tidb_version()").Scan(&v); err != nil {
		return fmt.Sprintf("query error: %v", err)
	}
	return v
}

func sanitizeFileName(s string) string {
	re := regexp.MustCompile(`[^A-Za-z0-9_\-]+`)
	s = strings.TrimSpace(s)
	s = strings.ReplaceAll(s, " ", "_")
	return re.ReplaceAllString(s, "")
}

func extractStatsMetaFromLog(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	lines := strings.Split(string(b), "\n")
	in := false
	out := make([]string, 0, 8)
	for _, line := range lines {
		if strings.HasPrefix(line, "[DB] STATS_META for DB=") {
			in = true
			out = append(out, line)
			continue
		}
		if in {
			if strings.TrimSpace(line) == "" {
				if len(out) > 0 {
					break
				}
				continue
			}
			if !strings.Contains(line, " | ") {
				break
			}
			out = append(out, line)
		}
	}
	if len(out) == 0 {
		return ""
	}
	return strings.Join(out, "\n")
}

// issueExistsForModelLabel returns true if any existing issue file under runsDir/issues
// matches the same Model + QueryLabel pair (ignoring scenario ratio/label).
func issueExistsForModelLabel(runsDir, model, queryLabel string) bool {
	root := filepath.Join(runsDir, "issues")
	if entries, err := os.ReadDir(root); err == nil {
		base := fmt.Sprintf("issue_%s_%s_", sanitizeFileName(model), sanitizeFileName(queryLabel))
		for _, e := range entries {
			name := e.Name()
			if e.IsDir() {
				// scan subdir files
				sub := filepath.Join(root, name)
				files, _ := os.ReadDir(sub)
				for _, f := range files {
					if !f.IsDir() && strings.HasPrefix(f.Name(), base) {
						return true
					}
				}
			} else {
				if strings.HasPrefix(name, base) {
					return true
				}
			}
		}
	}
	return false
}

func writeIssueFiles(runsDir string, dbCfg itypes.DBConfig, label string, rep reportJSON, statsMeta string) error {
	ts := time.Now().Format("20060102_150405")
	outDir := filepath.Join(runsDir, "issues", ts)
	if err := os.MkdirAll(outDir, 0755); err != nil {
		return err
	}
	version := fetchTiDBVersion(dbCfg)
	for _, r := range rep.Results {
		if !(r.EstimationErrorRatio >= 10 && r.EstimationErrorValue >= 1000) {
			continue
		}
		// Skip if an issue with same Model + QueryLabel already exists
		if issueExistsForModelLabel(runsDir, r.Model, r.QueryLabel) {
			fmt.Printf("[Issue] Skip duplicate (same Model+QueryLabel): %s / %s\n", r.Model, r.QueryLabel)
			continue
		}
		// Title: Model-query label-sql label+变更率百分比
		// 这里将 sql label 取为场景标签（如 20/50/80），以“xx%”形式拼接
		title := fmt.Sprintf("%s-%s-%s%%", r.Model, r.QueryLabel, label)
		// normalize plan replayer path
		pr := normalizePlanReplayerPath(r.PlanReplayerLink)
		name := fmt.Sprintf("issue_%s_%s_%s.md", sanitizeFileName(r.Model), sanitizeFileName(r.QueryLabel), label)
		path := filepath.Join(outDir, name)
		body := &strings.Builder{}
		fmt.Fprintf(body, "title: %s\n\n", title)
		fmt.Fprintf(body, "1. Minimal reproduce step (Required)\n")
		fmt.Fprintf(body, "   drop database if exists %s;\n", dbCfg.DBName)
		if pr != "" {
			fmt.Fprintf(body, "   plan replayer load '%s'\n", pr)
		} else {
			fmt.Fprintf(body, "   -- plan replayer file not available\n")
		}
		fmt.Fprintf(body, "   %s\n\n", r.Query)
		fmt.Fprintf(body, "   show stats_meta;\n")
		if statsMeta != "" {
			fmt.Fprintf(body, "   ```\n")
			for _, line := range strings.Split(statsMeta, "\n") {
				fmt.Fprintf(body, "   %s\n", line)
			}
			fmt.Fprintf(body, "   ```\n\n")
		} else {
			fmt.Fprintf(body, "   -- show stats_meta output not found in run logs\n\n")
		}
		fmt.Fprintf(body, "2. What did you expect to see? (Required)\n")
		fmt.Fprintf(body, "   Est Error Ratio<10 and Est Error Value<1000\n\n")
		fmt.Fprintf(body, "3. What did you see instead (Required)\n")
		fmt.Fprintf(body, "   Est Error Ratio: %.2f\n", r.EstimationErrorRatio)
		fmt.Fprintf(body, "   Est Error Value: %.2f\n", r.EstimationErrorValue)
		fmt.Fprintf(body, "   Explain Plan:\n%s\n\n", r.Explain)
		fmt.Fprintf(body, "4. What is your TiDB version? (Required)\n")
		fmt.Fprintf(body, "   %s\n", version)
		if err := os.WriteFile(path, []byte(body.String()), 0644); err != nil {
			return err
		}
		fmt.Printf("[Issue] Generated: %s\n", path)
	}
	return nil
}

// parseRulesLabels extracts query labels from rules content.
// Heuristics: capture backticked labels after "Query Label" or occurrences of "LABEL: <name>".
func parseRulesLabels(content string) []string {
	labels := map[string]struct{}{}
	// Query Label: `name`
	reBacktick := regexp.MustCompile(`(?i)Query\s*Label\s*[:：]\s*` + "`" + `([^` + "`" + `]+)` + "`")
	for _, m := range reBacktick.FindAllStringSubmatch(content, -1) {
		labels[strings.TrimSpace(m[1])] = struct{}{}
	}
	// /* LABEL: name */ examples in text
	reLabel := regexp.MustCompile(`(?i)LABEL\s*[:：]\s*([A-Za-z0-9 _\-]+)`) // broad match
	for _, m := range reLabel.FindAllStringSubmatch(content, -1) {
		l := strings.TrimSpace(m[1])
		if l != "" {
			labels[l] = struct{}{}
		}
	}
	out := make([]string, 0, len(labels))
	for k := range labels {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func loadRulesState(path string) (rulesState, error) {
	var st rulesState
	b, err := os.ReadFile(path)
	if err != nil {
		return st, err
	}
	if err := json.Unmarshal(b, &st); err != nil {
		return st, err
	}
	return st, nil
}

func saveRulesState(path string, st rulesState) error {
	st.UpdatedAt = time.Now().Format(time.RFC3339)
	return writeJSON(path, st)
}

// applyRuleChanges reads SQLGEN_RULES.md-derived labels and ensures
// internal/query/querybuilder.go contains corresponding rule generators.
func applyRuleChanges(repoRoot string, labels []string) error {
	lower := map[string]struct{}{}
	for _, l := range labels {
		lower[strings.ToLower(strings.TrimSpace(l))] = struct{}{}
	}
	// Example: auto-implement "int include hole" under Holes model block
	if _, ok := lower["int include hole"]; ok {
		if err := ensureIntIncludeHoleRule(repoRoot, "int include hole", 10); err != nil {
			return err
		}
	}
	return nil
}

func ensureIntIncludeHoleRule(repoRoot string, label string, margin int) error {
	qbPath := filepath.Join(repoRoot, "internal", "query", "querybuilder.go")
	b, err := os.ReadFile(qbPath)
	if err != nil {
		return err
	}
	content := string(b)
	// Already exists?
	if strings.Contains(strings.ToLower(content), "/* label: "+strings.ToLower(label)+" */") || strings.Contains(content, "/* LABEL: int include hole */") {
		return nil
	}
	// Find insertion anchor after "int across hole" rule
	anchor := "/* LABEL: int across hole */"
	pos := strings.Index(content, anchor)
	if pos < 0 {
		// fallback: after Holes Specific header
		anchor2 := "-- [Int] Holes Specific Queries"
		pos = strings.Index(content, anchor2)
		if pos < 0 {
			// If we cannot find any anchor, skip silently
			return nil
		}
		// move to end of that line
		if i := strings.Index(content[pos:], "\n"); i >= 0 {
			pos = pos + i + 1
		}
	} else {
		// advance to end of the append line (first newline after anchor occurrence)
		if i := strings.Index(content[pos:], "\n"); i >= 0 {
			pos = pos + i + 1
		}
	}
	if margin <= 0 {
		margin = 10
	}
	snippet := "\n\t\t\t// New: int include hole (outside the hole interval with margins)\n" +
		fmt.Sprintf("\t\t\tleft := holeStart - %d\n", margin) +
		fmt.Sprintf("\t\t\tright := holeEnd + %d\n", margin) +
		"\t\t\tsqls = append(sqls, fmt.Sprintf(\"/* LABEL: %s */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s < %d OR %s > %d\", \n" +
		fmt.Sprintf("\t\t\t\t\t\t\t\t\t\t\t\t\"%s\", tableName, tableName, colInt, left, colInt, right))\n", label)
	newContent := content[:pos] + snippet + content[pos:]
	return os.WriteFile(qbPath, []byte(newContent), 0644)
}

// YAML DSL structures (JSON subset of YAML 1.2)
type RuleDoc struct {
	Rules []RuleSpec `json:"rules"`
}
type RuleSpec struct {
	Label     string                 `json:"label"`
	Type      string                 `json:"type"`
	AppliesTo []string               `json:"applies_to"` // model types or names; optional
	Params    map[string]interface{} `json:"params"`
}

func toIntDefault(m map[string]interface{}, key string, def int) int {
	if m == nil {
		return def
	}
	if v, ok := m[key]; ok {
		switch x := v.(type) {
		case float64:
			return int(x)
		case int:
			return x
		case string:
			var d int
			fmt.Sscanf(x, "%d", &d)
			if d != 0 {
				return d
			}
		}
	}
	return def
}

func applyRuleChangesYAML(repoRoot string, doc RuleDoc) error {
	for _, r := range doc.Rules {
		t := strings.ToLower(strings.TrimSpace(r.Type))
		switch t {
		case "holes_int_include", "int_include_hole":
			margin := toIntDefault(r.Params, "margin", 10)
			label := r.Label
			if label == "" {
				label = "int include hole"
			}
			if err := ensureIntIncludeHoleRule(repoRoot, label, margin); err != nil {
				return err
			}
		case "datetime_include_hole", "holes_datetime_include":
			label := r.Label
			if label == "" {
				label = "datetime include hole"
			}
			if err := ensureDatetimeIncludeHoleRule(repoRoot, label); err != nil {
				return err
			}
		case "lots_of_in":
			// already implemented in generator; nothing to inject
			fmt.Println("[Rules] lots_of_in recognized (already supported)")
		case "int_out_of_bound", "int_point_lookup", "int_range_scan",
			"int_last_histogram", "int_first_histogram",
			"string_out_of_bound", "string_point_lookup", "string_range_scan",
			"string_last_histogram", "string_first_histogram",
			"datetime_out_of_bound", "datetime_point_lookup", "datetime_range_scan",
			"datetime_last_histogram", "datetime_first_histogram",
			"holes_datetime_in", "holes_datetime_across",
			"mixed_condition":
			// These rules are already present in the generator by default
			fmt.Printf("[Rules] %s recognized (already supported)\n", t)
		default:
			fmt.Printf("[Rules] Unhandled rule type: %s (label=%s)\n", r.Type, r.Label)
		}
	}
	return nil
}

func ensureDatetimeIncludeHoleRule(repoRoot string, label string) error {
	qbPath := filepath.Join(repoRoot, "internal", "query", "querybuilder.go")
	b, err := os.ReadFile(qbPath)
	if err != nil {
		return err
	}
	content := string(b)
	// Already exists?
	low := strings.ToLower(content)
	if strings.Contains(low, "/* label: "+strings.ToLower(label)+" */") || strings.Contains(low, "/* label: datetime include hole */") {
		return nil
	}
	// Insert inside Holes datetime block, after across hole if possible
	anchor := "/* LABEL: datetime across hole */"
	pos := strings.Index(content, anchor)
	if pos < 0 {
		anchor2 := "-- [Datetime] Holes Specific Queries"
		pos = strings.Index(content, anchor2)
		if pos < 0 {
			return nil
		}
		if i := strings.Index(content[pos:], "\n"); i >= 0 {
			pos = pos + i + 1
		}
	} else {
		if i := strings.Index(content[pos:], "\n"); i >= 0 {
			pos = pos + i + 1
		}
	}
	// Use existing variables dhStart, dhEnd, offset inside that block
	snippet := "\n\t\t\t// New: datetime include hole (expanded interval around hole)\n" +
		"\t\t\tincludeStart := dhStart.Add(-offset).Format(\"2006-01-02\")\n" +
		"\t\t\tincludeEnd := dhEnd.Add(offset).Format(\"2006-01-02\")\n" +
		"\t\t\tsqls = append(sqls, fmt.Sprintf(\"/* LABEL: %s */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s > '%s' AND %s < '%s'\", \n" +
		fmt.Sprintf("\t\t\t\t\t\t\t\t\t\t\t\t\"%s\", tableName, tableName, colDt, includeStart, colDt, includeEnd))\n", label)
	newContent := content[:pos] + snippet + content[pos:]
	return os.WriteFile(qbPath, []byte(newContent), 0644)
}

func ensureIntOutOfBound2Rule(repoRoot, label string, delta int, side string) error {
	qbPath := filepath.Join(repoRoot, "internal", "query", "querybuilder.go")
	b, err := os.ReadFile(qbPath)
	if err != nil {
		return err
	}
	content := string(b)
	low := strings.ToLower(content)
	if strings.Contains(low, "/* label: "+strings.ToLower(label)+" */") {
		return nil
	}
	// insert near "int out of bound" anchor
	anchor := "/* LABEL: int out of bound */"
	pos := strings.Index(content, anchor)
	if pos < 0 {
		// fallback to int block start (after baseMinInt/baseMaxInt area)
		anchor2 := "// Use base min/max for histogram-boundary queries"
		pos = strings.Index(content, anchor2)
		if pos < 0 {
			return nil
		}
		if i := strings.Index(content[pos:], "\n"); i >= 0 {
			pos = pos + i + 1
		}
	} else {
		if i := strings.Index(content[pos:], "\n"); i >= 0 {
			pos = pos + i + 1
		}
	}
	if delta <= 0 {
		delta = 2000
	}
	// construct snippet depending on side
	mkUpper := func() string {
		return "\n\t\t\t// New: extended out of bound (upper)\n" +
			fmt.Sprintf("\t\t\tvalOB2U := maxInt + %d\n", delta) +
			"\t\t\tsqls = append(sqls, fmt.Sprintf(\"/* LABEL: %s */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = %d\", \n" +
			fmt.Sprintf("\t\t\t\t\t\t\t\t\t\t\t\t\"%s\", tableName, tableName, colInt, valOB2U))\n", label)
	}
	mkLower := func() string {
		return "\n\t\t\t// New: extended out of bound (lower)\n" +
			fmt.Sprintf("\t\t\tvalOB2L := minInt - %d\n", delta) +
			"\t\t\tsqls = append(sqls, fmt.Sprintf(\"/* LABEL: %s */ SELECT /*+ IGNORE_INDEX(%s PRIMARY) */ 1 FROM %s WHERE %s = %d\", \n" +
			fmt.Sprintf("\t\t\t\t\t\t\t\t\t\t\t\t\"%s\", tableName, tableName, colInt, valOB2L))\n", label)
	}
	snippet := ""
	switch side {
	case "lower":
		snippet = mkLower()
	case "both":
		snippet = mkUpper() + mkLower()
	default: // upper
		snippet = mkUpper()
	}
	newContent := content[:pos] + snippet + content[pos:]
	return os.WriteFile(qbPath, []byte(newContent), 0644)
}

// detectAndApplyRules returns true when SQLGEN_RULES.md changed since last run.
func detectAndApplyRules(repoRoot, rulesPath, statePath string, logNoChange bool) (bool, error) {
	if _, err := os.Stat(rulesPath); err != nil {
		fmt.Printf("[Rules] %s not found, skipping.\n", rulesPath)
		return false, nil
	}
	// load previous state
	var prev rulesState
	if b, err := os.ReadFile(statePath); err == nil {
		_ = json.Unmarshal(b, &prev)
	}
	// current content
	absRules := rulesPath
	if !filepath.IsAbs(absRules) {
		if wd, err := os.Getwd(); err == nil {
			absRules = filepath.Join(wd, rulesPath)
		}
	}
	sha, _ := fileSHA1(rulesPath)
	content, _ := os.ReadFile(rulesPath)
	curLabels := parseRulesLabels(string(content))
	info, _ := os.Stat(rulesPath)
	mtime := ""
	if info != nil {
		mtime = info.ModTime().Format(time.RFC3339)
	}
	fmt.Printf("[Rules] %s (abs=%s) mtime=%s sha1=%s, labels=%v\n", rulesPath, absRules, mtime, sha, curLabels)
	if sha == prev.FileSHA1 {
		if logNoChange {
			fmt.Println("[Rules] No change detected (sha1 unchanged).")
		}
		return false, nil
	}
	// diff labels for logging
	old := map[string]struct{}{}
	for _, l := range prev.Labels {
		old[l] = struct{}{}
	}
	newly := []string{}
	for _, l := range curLabels {
		if _, ok := old[l]; !ok {
			newly = append(newly, l)
		}
	}
	if len(newly) > 0 {
		fmt.Printf("[Rules] New labels detected: %v\n", newly)
	} else {
		fmt.Println("[Rules] File content changed but no new labels detected.")
	}
	// try applying recognized rules (label-based quick path)
	if err := applyRuleChanges(repoRoot, curLabels); err != nil {
		fmt.Println("[Rules] apply label-based changes error:", err)
	}
	// parse fenced YAML blocks for structured rules (subset parser without external deps)
	yamlDocs := []RuleDoc{}
	blocks := extractYAMLBlocks(string(content))
	for _, blk := range blocks {
		if d, err := parseRulesYAMLSubset(blk); err == nil && len(d.Rules) > 0 {
			yamlDocs = append(yamlDocs, d)
		} else if err != nil {
			fmt.Println("[Rules] YAML parse warning:", err)
		}
	}
	for _, d := range yamlDocs {
		if err := applyRuleChangesYAML(repoRoot, d); err != nil {
			fmt.Println("[Rules] apply YAML changes error:", err)
		}
	}
	// persist new state
	_ = os.MkdirAll(filepath.Dir(statePath), 0755)
	_ = saveRulesState(statePath, rulesState{FileSHA1: sha, Labels: curLabels})
	// hints (print once only when changed)
	if strings.Contains(strings.ToLower(string(content)), "lots of in") {
		fmt.Println("[Rules] Recognized directive: lots of IN (already enabled in generator)")
	}
	return true, nil
}

// extractYAMLBlocks scans Markdown content and returns a list of fenced YAML code blocks
// delimited by ```yaml ... ```
func extractYAMLBlocks(md string) []string {
	res := []string{}
	lines := strings.Split(md, "\n")
	in := false
	var buf []string
	for _, ln := range lines {
		trim := strings.TrimSpace(ln)
		if !in {
			if strings.HasPrefix(trim, "```yaml") || strings.HasPrefix(trim, "```yml") {
				in = true
				buf = []string{}
			}
			continue
		}
		if strings.HasPrefix(trim, "```") {
			// end
			in = false
			if len(buf) > 0 {
				res = append(res, strings.Join(buf, "\n"))
			}
			buf = nil
			continue
		}
		buf = append(buf, ln)
	}
	return res
}

// parseRulesYAMLSubset parses a limited YAML subset for RuleDoc:
// rules:
//   - label: "..."
//     type: some_type
//     applies_to: [holes, skew]
//     params:
//       margin: 10
//       side: upper
func parseRulesYAMLSubset(yamlText string) (RuleDoc, error) {
	var doc RuleDoc
	lines := strings.Split(yamlText, "\n")
	inRules := false
	var cur *RuleSpec
	inParams := false
	paramsIndent := 0
	keyVal := func(s string) (string, string, bool) {
		idx := strings.Index(s, ":")
		if idx < 0 {
			return "", "", false
		}
		k := strings.TrimSpace(s[:idx])
		v := strings.TrimSpace(s[idx+1:])
		return k, v, true
	}
	parseScalar := func(v string) interface{} {
		v = strings.TrimSpace(v)
		if v == "" {
			return ""
		}
		if (strings.HasPrefix(v, "\"") && strings.HasSuffix(v, "\"")) || (strings.HasPrefix(v, "'") && strings.HasSuffix(v, "'")) {
			return strings.Trim(v, "\"'")
		}
		if strings.HasPrefix(v, "[") && strings.HasSuffix(v, "]") {
			inner := strings.TrimSpace(v[1 : len(v)-1])
			if inner == "" {
				return []string{}
			}
			parts := strings.Split(inner, ",")
			out := make([]string, 0, len(parts))
			for _, p := range parts {
				p = strings.TrimSpace(p)
				p = strings.Trim(p, "\"'")
				if p != "" {
					out = append(out, p)
				}
			}
			return out
		}
		var i int
		if _, err := fmt.Sscanf(v, "%d", &i); err == nil {
			return i
		}
		return v
	}
	indentOf := func(s string) int {
		n := 0
		for _, ch := range s {
			if ch == ' ' {
				n++
			} else {
				break
			}
		}
		return n
	}
	for _, raw := range lines {
		line := raw
		trim := strings.TrimSpace(line)
		if trim == "" || strings.HasPrefix(trim, "#") {
			continue
		}
		if !inRules {
			if strings.HasPrefix(trim, "rules:") {
				inRules = true
			}
			continue
		}
		if strings.HasPrefix(trim, "- ") {
			if cur != nil {
				doc.Rules = append(doc.Rules, *cur)
			}
			cur = &RuleSpec{Params: map[string]interface{}{}, AppliesTo: []string{}}
			inParams = false
			continue
		}
		if cur == nil {
			continue
		}
		if strings.HasPrefix(trim, "params:") {
			inParams = true
			paramsIndent = indentOf(line)
			continue
		}
		if inParams {
			ind := indentOf(line)
			if ind <= paramsIndent {
				inParams = false
			} else {
				if k, v, ok := keyVal(strings.TrimSpace(line)); ok {
					cur.Params[k] = parseScalar(v)
					continue
				}
			}
		}
		if k, v, ok := keyVal(trim); ok {
			switch strings.ToLower(k) {
			case "label":
				if s, ok2 := parseScalar(v).(string); ok2 {
					cur.Label = s
				}
			case "type":
				if s, ok2 := parseScalar(v).(string); ok2 {
					cur.Type = s
				}
			case "applies_to":
				if arr, ok2 := parseScalar(v).([]string); ok2 {
					cur.AppliesTo = arr
				}
			}
		}
	}
	if cur != nil {
		doc.Rules = append(doc.Rules, *cur)
	}
	return doc, nil
}

func main() {
	cfgPath := flag.String("config", "config.json", "Base config file")
	dbCfgPath := flag.String("db-config", "db_config.json", "DB config file")
	rulesPath := flag.String("rules", "SQLGEN_RULES.md", "SQL generation prompt file")
	toolOut := flag.String("tool-output-dir", "output", "det-tool outputs directory")
	runsOut := flag.String("out", "runs", "run-scenarios output directory")
	ratios := flag.String("ratios", "0.2,0.5,0.8", "ratios for scenarios")
	watch := flag.Bool("watch", false, "Watch SQLGEN_RULES.md and run on changes")
	watchInterval := flag.Int("watch-interval", 30, "Watch interval seconds when --watch is set")
	flag.Parse()

	repoRoot, _ := os.Getwd()

	// Helper to run full pipeline once
	runOnce := func() {
		// Build
		if err := buildBinaries(repoRoot); err != nil {
			fmt.Println("[Build] Error:", err)
			return
		}
		// Run scenarios
		if err := runScenarios(repoRoot, *cfgPath, *dbCfgPath, *toolOut, *runsOut, *ratios); err != nil {
			fmt.Println("[Run] Error:", err)
			return
		}
		// Generate issues
		var dbc itypes.DBConfig
		_ = readJSON(*dbCfgPath, &dbc)
		entries, _ := os.ReadDir(*runsOut)
		scenarioLabels := []string{}
		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			name := e.Name()
			if strings.HasPrefix(name, "report_") && strings.HasSuffix(name, ".json") {
				lab := strings.TrimSuffix(strings.TrimPrefix(name, "report_"), ".json")
				scenarioLabels = append(scenarioLabels, lab)
			}
		}
		sort.Strings(scenarioLabels)
		for _, lab := range scenarioLabels {
			path := filepath.Join(*runsOut, fmt.Sprintf("report_%s.json", lab))
			var rep reportJSON
			if err := readJSON(path, &rep); err != nil {
				fmt.Printf("[Issue] parse %s error: %v\n", path, err)
				continue
			}
			logPath := filepath.Join(*runsOut, fmt.Sprintf("scenario_%s.log", lab))
			statsMeta := extractStatsMetaFromLog(logPath)
			if err := writeIssueFiles(*runsOut, dbc, lab, rep, statsMeta); err != nil {
				fmt.Printf("[Issue] write issues for %s error: %v\n", lab, err)
			}
		}
	}

	statePath := filepath.Join(".automation", "rules_state.json")
	// Detect and apply rule changes once
	changed, _ := detectAndApplyRules(repoRoot, *rulesPath, statePath, true)
	if !*watch {
		// One-shot mode: always run once (build + scenarios + issues)
		runOnce()
		return
	}
	// Watch mode: only run when rules changed
	if changed {
		runOnce()
	} else {
		fmt.Println("[Watch] Initial state: no change, waiting...")
	}
	interval := *watchInterval
	if interval <= 0 {
		interval = 30
	}
	for {
		time.Sleep(time.Duration(interval) * time.Second)
		changed, _ := detectAndApplyRules(repoRoot, *rulesPath, statePath, false)
		if changed {
			runOnce()
		}
	}
}
