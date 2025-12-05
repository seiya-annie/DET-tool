# DET-Tool Makefile

# Variables
BINARY_NAME=det-tool
GO=go
GOFLAGS=-v
BUILD_DIR=build
DIST_DIR=dist

# Version information
VERSION?=1.0.0
BUILD_TIME=$(shell date -u '+%Y-%m-%d_%H:%M:%S')
GIT_COMMIT=$(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
LDFLAGS=-ldflags "-X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME) -X main.GitCommit=$(GIT_COMMIT)"

# Default target
.PHONY: all
all: clean build

# Build the binary
.PHONY: build
build:
	@echo "Building $(BINARY_NAME)..."
	@mkdir -p $(BUILD_DIR)
	$(GO) build $(GOFLAGS) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME) .
	@echo "Build complete: $(BUILD_DIR)/$(BINARY_NAME)"

# Build for multiple platforms
.PHONY: build-all
build-all:
	@echo "Building for multiple platforms..."
	@mkdir -p $(DIST_DIR)
	
	# Linux AMD64
	GOOS=linux GOARCH=amd64 $(GO) build $(GOFLAGS) $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)-linux-amd64 .
	
	# Linux ARM64
	GOOS=linux GOARCH=arm64 $(GO) build $(GOFLAGS) $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)-linux-arm64 .
	
	# Darwin AMD64 (macOS)
	GOOS=darwin GOARCH=amd64 $(GO) build $(GOFLAGS) $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)-darwin-amd64 .
	
	# Darwin ARM64 (macOS M1)
	GOOS=darwin GOARCH=arm64 $(GO) build $(GOFLAGS) $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)-darwin-arm64 .
	
	# Windows AMD64
	GOOS=windows GOARCH=amd64 $(GO) build $(GOFLAGS) $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)-windows-amd64.exe .
	
	@echo "Multi-platform build complete"

# Run tests
.PHONY: test
test:
	@echo "Running tests..."
	$(GO) test -v ./...

# Run tests with coverage
.PHONY: test-coverage
test-coverage:
	@echo "Running tests with coverage..."
	$(GO) test -v -coverprofile=coverage.out ./...
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

# Run benchmarks
.PHONY: bench
bench:
	@echo "Running benchmarks..."
	$(GO) test -bench=. -benchmem ./...

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR) $(DIST_DIR)
	@rm -f coverage.out coverage.html
	@rm -f *.log *.csv *.sql *.html
	$(GO) clean

# Install dependencies
.PHONY: deps
deps:
	@echo "Installing dependencies..."
	$(GO) mod download
	$(GO) mod tidy

# Update dependencies
.PHONY: deps-update
deps-update:
	@echo "Updating dependencies..."
	$(GO) get -u ./...
	$(GO) mod tidy

# Run the application
.PHONY: run
run: build
	@echo "Running $(BINARY_NAME)..."
	./$(BUILD_DIR)/$(BINARY_NAME) --all

# Run with example configuration
.PHONY: run-example
run-example: build
	@echo "Running with example configuration..."
	./$(BUILD_DIR)/$(BINARY_NAME) --config config.json --db-config db_config.json --all

# Generate example configuration files
.PHONY: gen-config
gen-config:
	@echo "Generating example configuration files..."
	@cp config.json.example config.json 2>/dev/null || echo "config.json.example not found, using current config.json"
	@cp db_config.json.example db_config.json 2>/dev/null || echo "db_config.json.example not found, using current db_config.json"

# Lint the code
.PHONY: lint
lint:
	@echo "Running linter..."
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run; \
	else \
		echo "golangci-lint not found, installing..."; \
		go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest; \
		golangci-lint run; \
	fi

# Format code
.PHONY: fmt
fmt:
	@echo "Formatting code..."
	$(GO) fmt ./...

# Vet code
.PHONY: vet
vet:
	@echo "Vetting code..."
	$(GO) vet ./...

# Security scan
.PHONY: security
security:
	@echo "Running security scan..."
	@if command -v gosec >/dev/null 2>&1; then \
		gosec ./...; \
	else \
		echo "gosec not found, installing..."; \
		go install github.com/securego/gosec/v2/cmd/gosec@latest; \
		gosec ./...; \
	fi

# Create release package
.PHONY: package
package: build-all
	@echo "Creating release packages..."
	@mkdir -p $(DIST_DIR)/releases
	
	# Create tar.gz packages for each platform
	@for file in $(DIST_DIR)/$(BINARY_NAME)-*; do \
		if [ -f "$$file" ]; then \
			base=$$(basename $$file); \
			tar -czf $(DIST_DIR)/releases/$$base.tar.gz -C $(DIST_DIR) $$base; \
			echo "Created $$base.tar.gz"; \
		fi \
	done
	
	# Copy configuration files
	@cp config.json $(DIST_DIR)/releases/config.json.example 2>/dev/null || true
	@cp db_config.json $(DIST_DIR)/releases/db_config.json.example 2>/dev/null || true
	@cp README.md $(DIST_DIR)/releases/ 2>/dev/null || true
	@cp LICENSE $(DIST_DIR)/releases/ 2>/dev/null || true
	
	@echo "Release packages created in $(DIST_DIR)/releases/"

# Docker build
.PHONY: docker-build
docker-build:
	@echo "Building Docker image..."
	docker build -t $(BINARY_NAME):$(VERSION) .
	docker tag $(BINARY_NAME):$(VERSION) $(BINARY_NAME):latest

# Docker run
.PHONY: docker-run
docker-run:
	@echo "Running Docker container..."
	docker run --rm -it \
		-v $$(pwd)/config.json:/app/config.json \
		-v $$(pwd)/db_config.json:/app/db_config.json \
		$(BINARY_NAME):latest --all

# Help
.PHONY: help
help:
	@echo "DET-Tool Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Clean and build the binary"
	@echo "  build        - Build the binary"
	@echo "  build-all    - Build for multiple platforms"
	@echo "  test         - Run tests"
	@echo "  test-coverage - Run tests with coverage"
	@echo "  bench        - Run benchmarks"
	@echo "  clean        - Clean build artifacts"
	@echo "  deps         - Install dependencies"
	@echo "  deps-update  - Update dependencies"
	@echo "  run          - Build and run the application"
	@echo "  run-example  - Run with example configuration"
	@echo "  gen-config   - Generate example configuration files"
	@echo "  lint         - Run linter"
	@echo "  fmt          - Format code"
	@echo "  vet          - Vet code"
	@echo "  security     - Run security scan"
	@echo "  package      - Create release packages"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  help         - Show this help message"

# Default make target
.DEFAULT_GOAL := help