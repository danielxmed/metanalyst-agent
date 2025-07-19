#!/bin/bash

# =====================================================
# METANALYST-AGENT POSTGRESQL SETUP SCRIPT
# =====================================================
# Este script automatiza a configuração completa do
# PostgreSQL para o sistema metanalyst-agent
# =====================================================

set -e  # Exit on any error

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Verificar se PostgreSQL está instalado
check_postgresql() {
    log "Verificando se PostgreSQL está instalado..."
    
    if command -v psql &> /dev/null; then
        success "PostgreSQL encontrado: $(psql --version)"
    else
        error "PostgreSQL não encontrado. Por favor, instale o PostgreSQL primeiro."
    fi
    
    if command -v createdb &> /dev/null; then
        success "Utilitários PostgreSQL encontrados"
    else
        error "Utilitários PostgreSQL não encontrados"
    fi
}

# Instalar PostgreSQL (Ubuntu/Debian)
install_postgresql_ubuntu() {
    log "Instalando PostgreSQL no Ubuntu/Debian..."
    
    sudo apt update
    sudo apt install -y postgresql postgresql-contrib postgresql-client
    
    # Iniciar e habilitar o serviço
    sudo systemctl start postgresql
    sudo systemctl enable postgresql
    
    success "PostgreSQL instalado com sucesso"
}

# Instalar PostgreSQL (CentOS/RHEL/Fedora)
install_postgresql_centos() {
    log "Instalando PostgreSQL no CentOS/RHEL/Fedora..."
    
    sudo dnf install -y postgresql postgresql-server postgresql-contrib
    
    # Inicializar o banco
    sudo postgresql-setup --initdb
    
    # Iniciar e habilitar o serviço
    sudo systemctl start postgresql
    sudo systemctl enable postgresql
    
    success "PostgreSQL instalado com sucesso"
}

# Configurar PostgreSQL
configure_postgresql() {
    log "Configurando PostgreSQL..."
    
    # Detectar versão do PostgreSQL
    PG_VERSION=$(psql --version | awk '{print $3}' | sed 's/\..*//')
    log "Versão do PostgreSQL detectada: $PG_VERSION"
    
    # Caminhos comuns para arquivos de configuração
    PG_CONFIG_PATHS=(
        "/etc/postgresql/$PG_VERSION/main"
        "/var/lib/pgsql/$PG_VERSION/data"
        "/usr/local/var/postgres"
        "/opt/homebrew/var/postgres"
    )
    
    PG_CONFIG_DIR=""
    for path in "${PG_CONFIG_PATHS[@]}"; do
        if [ -d "$path" ]; then
            PG_CONFIG_DIR="$path"
            break
        fi
    done
    
    if [ -z "$PG_CONFIG_DIR" ]; then
        warning "Diretório de configuração do PostgreSQL não encontrado automaticamente"
        log "Você pode precisar configurar manualmente o pg_hba.conf e postgresql.conf"
    else
        log "Diretório de configuração encontrado: $PG_CONFIG_DIR"
        
        # Backup dos arquivos de configuração
        sudo cp "$PG_CONFIG_DIR/pg_hba.conf" "$PG_CONFIG_DIR/pg_hba.conf.backup.$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
        sudo cp "$PG_CONFIG_DIR/postgresql.conf" "$PG_CONFIG_DIR/postgresql.conf.backup.$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
        
        # Configurar autenticação
        log "Configurando autenticação..."
        if [ -f "$PG_CONFIG_DIR/pg_hba.conf" ]; then
            # Permitir conexões locais com senha
            sudo sed -i "s/local   all             all                                     peer/local   all             all                                     md5/" "$PG_CONFIG_DIR/pg_hba.conf" 2>/dev/null || true
            sudo sed -i "s/local   all             all                                     ident/local   all             all                                     md5/" "$PG_CONFIG_DIR/pg_hba.conf" 2>/dev/null || true
        fi
        
        # Reiniciar PostgreSQL para aplicar configurações
        sudo systemctl restart postgresql || sudo service postgresql restart
    fi
}

# Criar usuário e banco de dados
create_database() {
    log "Criando usuário e banco de dados..."
    
    # Definir variáveis
    DB_NAME="${DB_NAME:-metanalysis}"
    DB_USER="${DB_USER:-metanalyst}"
    DB_PASSWORD="${DB_PASSWORD:-$(openssl rand -base64 32)}"
    
    # Criar usuário se não existir
    sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';" 2>/dev/null || {
        log "Usuário $DB_USER já existe, atualizando senha..."
        sudo -u postgres psql -c "ALTER USER $DB_USER WITH PASSWORD '$DB_PASSWORD';"
    }
    
    # Criar banco se não existir
    sudo -u postgres createdb -O "$DB_USER" "$DB_NAME" 2>/dev/null || {
        log "Banco $DB_NAME já existe"
    }
    
    # Conceder privilégios
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
    
    success "Banco de dados configurado:"
    echo "  Database: $DB_NAME"
    echo "  User: $DB_USER"
    echo "  Password: $DB_PASSWORD"
    
    # Salvar credenciais em arquivo
    cat > .env.db << EOF
# Database credentials generated by setup script
DATABASE_URL=postgresql://$DB_USER:$DB_PASSWORD@localhost:5432/$DB_NAME
DB_NAME=$DB_NAME
DB_USER=$DB_USER
DB_PASSWORD=$DB_PASSWORD
EOF
    
    success "Credenciais salvas em .env.db"
}

# Executar script SQL
run_sql_setup() {
    log "Executando script de configuração SQL..."
    
    if [ ! -f "scripts/setup_database.sql" ]; then
        error "Arquivo scripts/setup_database.sql não encontrado"
    fi
    
    # Executar como o usuário da aplicação
    PGPASSWORD="$DB_PASSWORD" psql -h localhost -U "$DB_USER" -d "$DB_NAME" -f scripts/setup_database.sql
    
    success "Script SQL executado com sucesso"
}

# Verificar conectividade
test_connection() {
    log "Testando conectividade com o banco..."
    
    PGPASSWORD="$DB_PASSWORD" psql -h localhost -U "$DB_USER" -d "$DB_NAME" -c "SELECT 'Conexão OK' as status;" > /dev/null
    
    success "Conexão com banco estabelecida com sucesso"
}

# Instalar extensões Python necessárias
install_python_deps() {
    log "Instalando dependências Python para PostgreSQL..."
    
    pip install psycopg2-binary || pip install psycopg2
    
    success "Dependências Python instaladas"
}

# Criar estrutura de diretórios
create_directories() {
    log "Criando estrutura de diretórios..."
    
    mkdir -p data/{vector_store,temp,backups}
    mkdir -p logs
    
    success "Diretórios criados"
}

# Função principal
main() {
    echo "=============================================="
    echo "    METANALYST-AGENT POSTGRESQL SETUP"
    echo "=============================================="
    echo
    
    # Parse argumentos
    INSTALL_PG=false
    AUTO_CONFIG=false
    SKIP_DEPS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --install-pg)
                INSTALL_PG=true
                shift
                ;;
            --auto-config)
                AUTO_CONFIG=true
                shift
                ;;
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --db-name)
                DB_NAME="$2"
                shift 2
                ;;
            --db-user)
                DB_USER="$2"
                shift 2
                ;;
            --db-password)
                DB_PASSWORD="$2"
                shift 2
                ;;
            -h|--help)
                echo "Uso: $0 [opções]"
                echo
                echo "Opções:"
                echo "  --install-pg      Instalar PostgreSQL automaticamente"
                echo "  --auto-config     Configurar PostgreSQL automaticamente"
                echo "  --skip-deps       Pular instalação de dependências Python"
                echo "  --db-name NAME    Nome do banco (default: metanalysis)"
                echo "  --db-user USER    Nome do usuário (default: metanalyst)"
                echo "  --db-password PWD Senha do usuário (default: gerada automaticamente)"
                echo "  -h, --help        Mostrar esta ajuda"
                echo
                exit 0
                ;;
            *)
                error "Opção desconhecida: $1"
                ;;
        esac
    done
    
    # Verificar sistema operacional
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
    else
        OS=$(uname -s)
    fi
    
    log "Sistema operacional detectado: $OS"
    
    # Instalar PostgreSQL se solicitado
    if [ "$INSTALL_PG" = true ]; then
        case $OS in
            ubuntu|debian)
                install_postgresql_ubuntu
                ;;
            centos|rhel|fedora)
                install_postgresql_centos
                ;;
            *)
                warning "Sistema operacional não suportado para instalação automática"
                warning "Por favor, instale o PostgreSQL manualmente"
                ;;
        esac
    fi
    
    # Verificar PostgreSQL
    check_postgresql
    
    # Configurar PostgreSQL se solicitado
    if [ "$AUTO_CONFIG" = true ]; then
        configure_postgresql
    fi
    
    # Criar estrutura de diretórios
    create_directories
    
    # Instalar dependências Python
    if [ "$SKIP_DEPS" = false ]; then
        install_python_deps
    fi
    
    # Criar banco e usuário
    create_database
    
    # Executar script SQL
    run_sql_setup
    
    # Testar conexão
    test_connection
    
    echo
    echo "=============================================="
    echo "           CONFIGURAÇÃO CONCLUÍDA!"
    echo "=============================================="
    echo
    echo "Próximos passos:"
    echo "1. Copie as credenciais de .env.db para seu .env principal"
    echo "2. Execute 'pip install -r requirements.txt' se ainda não fez"
    echo "3. Inicie o metanalyst-agent!"
    echo
    echo "Comandos úteis:"
    echo "- Conectar ao banco: psql -h localhost -U $DB_USER -d $DB_NAME"
    echo "- Ver estatísticas: psql -h localhost -U $DB_USER -d $DB_NAME -c \"SELECT * FROM database_stats();\""
    echo "- Backup: pg_dump -h localhost -U $DB_USER -d $DB_NAME > backup.sql"
    echo
    success "Setup completo!"
}

# Executar se chamado diretamente
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi