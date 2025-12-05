# DET-Tool Dockerfile

# Build stage
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git make

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o det-tool .

# Runtime stage
FROM alpine:latest

# Install runtime dependencies
RUN apk --no-cache add ca-certificates tzdata mysql-client

# Create non-root user
RUN addgroup -g 1000 -S detuser && \
    adduser -u 1000 -S detuser -G detuser

# Set working directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/det-tool /app/det-tool

# Copy default configuration files
COPY --from=builder /app/config.json /app/config.json.example
COPY --from=builder /app/db_config.json /app/db_config.json.example
COPY --from=builder /app/README.md /app/README.md

# Create directories for data and reports
RUN mkdir -p /app/data /app/reports && \
    chown -R detuser:detuser /app

# Switch to non-root user
USER detuser

# Set environment variables
ENV DET_TOOL_HOME=/app
ENV DET_CONFIG_PATH=/app/config.json
ENV DET_DB_CONFIG_PATH=/app/db_config.json

# Expose volume for configuration and data
VOLUME ["/app/config", "/app/data", "/app/reports"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /app/det-tool --help >/dev/null 2>&1 || exit 1

# Default command
ENTRYPOINT ["/app/det-tool"]

# Default arguments (can be overridden)
CMD ["--help"]

# Labels
LABEL maintainer="DET-Tool Team"
LABEL version="1.0.0"
LABEL description="Database Estimation Testing Tool"
LABEL org.opencontainers.image.title="DET-Tool"
LABEL org.opencontainers.image.description="Database Estimation Testing Tool for performance analysis"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.licenses="MIT"