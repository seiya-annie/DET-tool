package types

type Config struct {
    Models []ModelConfig `json:"models"`
}

type ModelConfig struct {
    Name        string                 `json:"name"`
    Description string                 `json:"description"`
    Type        string                 `json:"type"`
    Params      map[string]interface{} `json:"params"`
    Incremental map[string]interface{} `json:"incremental"`
}

type DBConfig struct {
    Host     string `json:"host"`
    Port     int    `json:"port"`
    User     string `json:"user"`
    Password string `json:"password"`
    DBName   string `json:"db_name"`
    Charset  string `json:"charset"`
}

type QueryResult struct {
    QueryID              int     `json:"query_id"`
    Query                string  `json:"query"`
    QueryLabel           string  `json:"query_label"`
    DurationMs           float64 `json:"duration_ms"`
    Explain              string  `json:"explain"`
    EstimationErrorValue float64 `json:"estimation_error_value"`
    EstimationErrorRatio float64 `json:"estimation_error_ratio"`
    RiskOperatorsCount   int     `json:"risk_operators_count"`
    Model                string  `json:"model"`
}

