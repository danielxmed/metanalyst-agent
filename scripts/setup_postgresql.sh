#!/bin/bash

# PostgreSQL Setup Script for Metanalyst-Agent
# This script sets up a PostgreSQL database ready for the metanalyst-agent system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Setting up PostgreSQL for Metanalyst-Agent${NC}"
echo "=================================================="

# Default configuration
DB_NAME="metanalysis"
DB_USER="metanalyst"
DB_PASSWORD="password"
DB_HOST="localhost"
DB_PORT="5432"

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo -e "${RED}❌ PostgreSQL is not installed. Please install PostgreSQL first.${NC}"
    echo "Installation commands:"
    echo "  macOS (with Homebrew): brew install postgresql"
    echo "  Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib"
    echo "  CentOS/RHEL: sudo yum install postgresql-server postgresql-contrib"
    exit 1
fi

echo -e "${GREEN}✅ PostgreSQL found${NC}"

# Check if PostgreSQL service is running
if ! pg_isready -h $DB_HOST -p $DB_PORT &> /dev/null; then
    echo -e "${YELLOW}⚠️  PostgreSQL service is not running. Starting it...${NC}"
    
    # Try to start PostgreSQL service
    if command -v brew &> /dev/null; then
        # macOS with Homebrew
        brew services start postgresql || true
    elif command -v systemctl &> /dev/null; then
        # Linux with systemd
        sudo systemctl start postgresql || true
    else
        echo -e "${RED}❌ Unable to start PostgreSQL automatically. Please start it manually.${NC}"
        exit 1
    fi
    
    # Wait a moment for service to start
    sleep 2
    
    if ! pg_isready -h $DB_HOST -p $DB_PORT &> /dev/null; then
        echo -e "${RED}❌ PostgreSQL service failed to start. Please check your PostgreSQL installation.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✅ PostgreSQL service is running${NC}"

# Create database user
echo -e "${BLUE}👤 Creating database user '${DB_USER}'...${NC}"
sudo -u postgres psql -c "CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASSWORD}';" 2>/dev/null || echo -e "${YELLOW}⚠️  User '${DB_USER}' may already exist${NC}"

# Grant privileges to user
echo -e "${BLUE}🔑 Granting privileges to user '${DB_USER}'...${NC}"
sudo -u postgres psql -c "ALTER USER ${DB_USER} CREATEDB;"

# Create database
echo -e "${BLUE}🗄️  Creating database '${DB_NAME}'...${NC}"
sudo -u postgres psql -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};" 2>/dev/null || echo -e "${YELLOW}⚠️  Database '${DB_NAME}' may already exist${NC}"

# Grant all privileges on database
echo -e "${BLUE}🔐 Granting all privileges on database '${DB_NAME}' to user '${DB_USER}'...${NC}"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};"

# Test connection
echo -e "${BLUE}🧪 Testing database connection...${NC}"
if PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1;" &> /dev/null; then
    echo -e "${GREEN}✅ Database connection successful!${NC}"
else
    echo -e "${RED}❌ Database connection failed${NC}"
    exit 1
fi

# Create .env file with database configuration
ENV_FILE="../.env"
echo -e "${BLUE}📝 Creating .env file...${NC}"

cat > $ENV_FILE << EOF
# Database Configuration for Metanalyst-Agent
DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}

# API Keys (please add your actual keys)
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: Uncomment and configure if needed
# REDIS_URL=redis://localhost:6379/0
# MONGODB_URL=mongodb://localhost:27017/metanalysis

# Agent Configuration
DEFAULT_RECURSION_LIMIT=100
DEFAULT_MAX_ARTICLES=50
DEFAULT_QUALITY_THRESHOLD=0.8

# Development Configuration
DEBUG=true
DEVELOPMENT_MODE=true
LOG_LEVEL=INFO
EOF

echo -e "${GREEN}✅ .env file created at ${ENV_FILE}${NC}"

# Install Python dependencies if requirements.txt exists
if [ -f "../requirements.txt" ]; then
    echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
    if command -v pip &> /dev/null; then
        pip install -r ../requirements.txt
        echo -e "${GREEN}✅ Python dependencies installed${NC}"
    else
        echo -e "${YELLOW}⚠️  pip not found. Please install Python dependencies manually:${NC}"
        echo "    pip install -r requirements.txt"
    fi
fi

# Initialize LangGraph tables
echo -e "${BLUE}🔧 Initializing LangGraph tables...${NC}"
python3 -c "
import os
import sys
sys.path.append('..')

try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.store.postgres import PostgresStore
    
    db_url = 'postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}'
    
    # Initialize checkpointer tables
    with PostgresSaver.from_conn_string(db_url) as checkpointer:
        checkpointer.setup()
    
    # Initialize store tables  
    with PostgresStore.from_conn_string(db_url) as store:
        store.setup()
    
    print('✅ LangGraph tables initialized successfully')
    
except Exception as e:
    print(f'❌ Error initializing LangGraph tables: {e}')
    print('This is normal if LangGraph dependencies are not installed yet.')
"

echo ""
echo -e "${GREEN}🎉 PostgreSQL setup completed successfully!${NC}"
echo "=================================================="
echo -e "${BLUE}Database Configuration:${NC}"
echo "  Host: $DB_HOST"
echo "  Port: $DB_PORT"
echo "  Database: $DB_NAME"
echo "  User: $DB_USER"
echo "  Password: $DB_PASSWORD"
echo ""
echo -e "${BLUE}Connection String:${NC}"
echo "  postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
echo ""
echo -e "${YELLOW}📋 Next Steps:${NC}"
echo "1. Add your OpenAI API key to the .env file"
echo "2. Add your Tavily API key to the .env file"
echo "3. Install Python dependencies: pip install -r requirements.txt"
echo "4. Run the metanalyst-agent: python -m metanalyst_agent.main"
echo ""
echo -e "${GREEN}✨ Your PostgreSQL database is ready for Metanalyst-Agent!${NC}"