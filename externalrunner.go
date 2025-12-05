package main

import (
	"fmt"
	"os/exec"
	"strings"
	"time"
)

// ExternalBenchRunner runs external benchmark tools
type ExternalBenchRunner struct {
	config DBConfig
}

// NewExternalBenchRunner creates a new ExternalBenchRunner
func NewExternalBenchRunner(config DBConfig) *ExternalBenchRunner {
	return &ExternalBenchRunner{config: config}
}

// PrepareData prepares data using external benchmark tool
func (ebr *ExternalBenchRunner) PrepareData(modelConfig ModelConfig) {
	modelType := modelConfig.Type
	if !strings.HasPrefix(modelType, "external_") {
		return
	}

	toolName := strings.Replace(modelType, "external_", "", 1)
	params := modelConfig.Params

	fmt.Printf("\n[External] Executing PREPARE for %s...\n", toolName)

	cmd := ebr.buildBaseCommand(toolName)
	cmd = append(cmd, "prepare")
	cmd = append(cmd, "--dropdata")

	// Add tool-specific parameters
	switch toolName {
	case "tpcc":
		if warehouses, ok := params["warehouses"].(float64); ok {
			cmd = append(cmd, "--warehouses", fmt.Sprintf("%.0f", warehouses))
		}
	case "tpch":
		if scaleFactor, ok := params["scale_factor"].(float64); ok {
			cmd = append(cmd, "--sf", fmt.Sprintf("%d", int(scaleFactor)))
		}
	}

	// Add extra arguments if specified
	if extraArgs, ok := params["extra_args"].(string); ok && extraArgs != "" {
		args := strings.Fields(extraArgs)
		cmd = append(cmd, args...)
	}

	ebr.runCommand(cmd)
}

// RunWorkload runs workload using external benchmark tool
func (ebr *ExternalBenchRunner) RunWorkload(modelConfig ModelConfig) {
	modelType := modelConfig.Type
	if !strings.HasPrefix(modelType, "external_") {
		return
	}

	toolName := strings.Replace(modelType, "external_", "", 1)
	baseParams := modelConfig.Params
	incremental := modelConfig.Incremental

	fmt.Printf("\n[External] Executing RUN (Incremental) for %s...\n", toolName)

	cmd := ebr.buildBaseCommand(toolName)
	cmd = append(cmd, "run")

	// Add tool-specific base parameters
	switch toolName {
	case "tpcc":
		if warehouses, ok := baseParams["warehouses"].(float64); ok {
			cmd = append(cmd, "--warehouses", fmt.Sprintf("%.0f", warehouses))
		}
		if time, ok := incremental["time"].(string); ok {
			cmd = append(cmd, "--time", time)
		}
		if threads, ok := incremental["threads"].(float64); ok {
			cmd = append(cmd, "--threads", fmt.Sprintf("%.0f", threads))
		}
	case "tpch":
		if scaleFactor, ok := baseParams["scale_factor"].(float64); ok {
			cmd = append(cmd, "--sf", fmt.Sprintf("%.1f", scaleFactor))
		}
		if queries, ok := incremental["queries"].([]interface{}); ok {
			queryStrings := make([]string, len(queries))
			for i, q := range queries {
				queryStrings[i] = fmt.Sprintf("%v", q)
			}
			cmd = append(cmd, "--queries", strings.Join(queryStrings, ","))
		}
	}

	ebr.runCommand(cmd)
}

// buildBaseCommand builds the base command for tiup bench
func (ebr *ExternalBenchRunner) buildBaseCommand(toolName string) []string {
	cmd := []string{
		"tiup", "bench", toolName,
		"--host", ebr.config.Host,
		"--port", fmt.Sprintf("%d", ebr.config.Port),
		"--user", ebr.config.User,
	}

	if ebr.config.Password != "" {
		cmd = append(cmd, "--password", ebr.config.Password)
	}

	if ebr.config.DBName != "" {
		cmd = append(cmd, "--db", ebr.config.DBName)
	}

	return cmd
}

// runCommand executes the command and handles output
func (ebr *ExternalBenchRunner) runCommand(cmd []string) {
	fmt.Printf("  Command: %s\n", strings.Join(cmd, " "))

	// Check if tiup is available
	if _, err := exec.LookPath("tiup"); err != nil {
		fmt.Println("  -> Error: 'tiup' command not found. Please install TiUP.")
		return
	}

	// Create command
	command := exec.Command(cmd[0], cmd[1:]...)
	
	// Set up output capture
	output, err := command.CombinedOutput()
	
	if err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			fmt.Printf("  -> External command failed with exit code %d\n", exitError.ExitCode())
		} else {
			fmt.Printf("  -> External command failed: %v\n", err)
		}
		
		if len(output) > 0 {
			fmt.Printf("  Output: %s\n", string(output))
		}
		return
	}

	fmt.Println("  -> External command finished successfully.")
	if len(output) > 0 {
		fmt.Printf("  Output: %s\n", string(output))
	}
}

// CheckTiUPAvailability checks if TiUP is available on the system
func (ebr *ExternalBenchRunner) CheckTiUPAvailability() bool {
	_, err := exec.LookPath("tiup")
	return err == nil
}

// GetTiUPVersion gets the version of TiUP if available
func (ebr *ExternalBenchRunner) GetTiUPVersion() (string, error) {
	cmd := exec.Command("tiup", "--version")
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(output)), nil
}

// InstallTiUP installs TiUP if not available
func (ebr *ExternalBenchRunner) InstallTiUP() error {
	fmt.Println("Installing TiUP...")
	
	// Download and install script
	installScript := `curl --proto '=https' --tlsv1.2 -sSf https://tiup-mirrors.pingcap.com/install.sh | sh`
	
	cmd := exec.Command("sh", "-c", installScript)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to install TiUP: %v\nOutput: %s", err, string(output))
	}

	fmt.Println("TiUP installed successfully.")
	fmt.Println("Please add TiUP to your PATH and restart the application.")
	return nil
}

// InstallBenchComponent installs the bench component for TiUP
func (ebr *ExternalBenchRunner) InstallBenchComponent() error {
	fmt.Println("Installing TiUP bench component...")
	
	cmd := exec.Command("tiup", "install", "bench")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to install bench component: %v\nOutput: %s", err, string(output))
	}

	fmt.Println("TiUP bench component installed successfully.")
	return nil
}

// RunCustomCommand runs a custom tiup bench command
func (ebr *ExternalBenchRunner) RunCustomCommand(toolName string, args []string) error {
	cmd := []string{"tiup", "bench", toolName}
	cmd = append(cmd, args...)
	
	fmt.Printf("Running custom command: %s\n", strings.Join(cmd, " "))
	
	command := exec.Command(cmd[0], cmd[1:]...)
	output, err := command.CombinedOutput()
	
	if err != nil {
		return fmt.Errorf("custom command failed: %v\nOutput: %s", err, string(output))
	}

	fmt.Printf("Custom command output:\n%s\n", string(output))
	return nil
}

// GetBenchmarkInfo gets information about available benchmarks
func (ebr *ExternalBenchRunner) GetBenchmarkInfo() map[string]string {
	return map[string]string{
		"tpcc": "TPC-C benchmark simulates a complete computing environment where a population of users executes transactions against a database. It includes 5 transaction types and measures performance in terms of transactions per minute (tpmC).",
		"tpch": "TPC-H benchmark consists of a suite of business oriented ad-hoc queries and concurrent data modifications. It measures the performance of decision support systems by executing complex queries against large datasets.",
	}
}

// ValidateConfig validates the model configuration for external benchmarks
func (ebr *ExternalBenchRunner) ValidateConfig(modelConfig ModelConfig) error {
	modelType := modelConfig.Type
	if !strings.HasPrefix(modelType, "external_") {
		return fmt.Errorf("model type %s is not an external benchmark", modelType)
	}

	toolName := strings.Replace(modelType, "external_", "", 1)
	validTools := []string{"tpcc", "tpch"}
	
	found := false
	for _, valid := range validTools {
		if toolName == valid {
			found = true
			break
		}
	}
	
	if !found {
		return fmt.Errorf("invalid external benchmark tool: %s. Valid tools are: %s", toolName, strings.Join(validTools, ", "))
	}

	// Validate required parameters
	params := modelConfig.Params
	switch toolName {
	case "tpcc":
		if _, ok := params["warehouses"]; !ok {
			return fmt.Errorf("tpcc benchmark requires 'warehouses' parameter")
		}
	case "tpch":
		if _, ok := params["scale_factor"]; !ok {
			return fmt.Errorf("tpch benchmark requires 'scale_factor' parameter")
		}
	}

	return nil
}

// EstimateDuration estimates the duration of a benchmark run
func (ebr *ExternalBenchRunner) EstimateDuration(modelConfig ModelConfig) time.Duration {
	modelType := modelConfig.Type
	if !strings.HasPrefix(modelType, "external_") {
		return 0
	}

	toolName := strings.Replace(modelType, "external_", "", 1)
	incremental := modelConfig.Incremental

	switch toolName {
	case "tpcc":
		if timeStr, ok := incremental["time"].(string); ok {
			// Parse duration string like "1m", "5m", "1h"
			if duration, err := time.ParseDuration(timeStr); err == nil {
				return duration
			}
		}
		return 1 * time.Minute // Default
	case "tpch":
		// TPC-H queries are typically faster
		return 30 * time.Second // Default
	default:
		return 1 * time.Minute
	}
}

// GetRequiredResources estimates required resources for the benchmark
func (ebr *ExternalBenchRunner) GetRequiredResources(modelConfig ModelConfig) map[string]string {
	modelType := modelConfig.Type
	if !strings.HasPrefix(modelType, "external_") {
		return map[string]string{}
	}

	toolName := strings.Replace(modelType, "external_", "", 1)
	params := modelConfig.Params

	resources := map[string]string{
		"cpu":     "2 cores minimum",
		"memory":  "4GB minimum",
		"storage": "10GB minimum",
	}

	switch toolName {
	case "tpcc":
		if warehouses, ok := params["warehouses"].(float64); ok {
			if warehouses > 10 {
				resources["cpu"] = "4 cores recommended"
				resources["memory"] = "8GB recommended"
				resources["storage"] = "50GB recommended"
			}
			if warehouses > 100 {
				resources["cpu"] = "8+ cores recommended"
				resources["memory"] = "16GB+ recommended"
				resources["storage"] = "200GB+ recommended"
			}
		}
	case "tpch":
		if scaleFactor, ok := params["scale_factor"].(float64); ok {
			if scaleFactor > 1 {
				resources["cpu"] = "4 cores recommended"
				resources["memory"] = "8GB recommended"
				resources["storage"] = "20GB recommended"
			}
			if scaleFactor > 10 {
				resources["cpu"] = "8+ cores recommended"
				resources["memory"] = "16GB+ recommended"
				resources["storage"] = "100GB+ recommended"
			}
		}
	}

	return resources
}

// MonitorExecution monitors the execution of external benchmarks
func (ebr *ExternalBenchRunner) MonitorExecution(modelConfig ModelConfig, progressChan chan<- string) {
	go func() {
		estimatedDuration := ebr.EstimateDuration(modelConfig)
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()

		startTime := time.Now()
		progressChan <- fmt.Sprintf("Started benchmark execution, estimated duration: %v", estimatedDuration)

		for {
			select {
			case <-ticker.C:
				elapsed := time.Since(startTime)
				if estimatedDuration > 0 {
					progress := float64(elapsed) / float64(estimatedDuration) * 100
					progressChan <- fmt.Sprintf("Progress: %.1f%% (elapsed: %v)", progress, elapsed)
				} else {
					progressChan <- fmt.Sprintf("Running... (elapsed: %v)", elapsed)
				}
			case <-time.After(estimatedDuration + 5*time.Minute):
				progressChan <- "Benchmark execution monitoring completed"
				return
			}
		}
	}()
}

// Cleanup cleans up after external benchmark execution
func (ebr *ExternalBenchRunner) Cleanup() error {
	fmt.Println("Cleaning up external benchmark resources...")
	
	// Clean up any temporary files or resources
	// This is a placeholder for cleanup operations
	
	return nil
}