#!/bin/bash

# Script de inicialização do Metanalyst-Agent CLI
# Usage: ./start_cli.sh [options]

echo "🔬 Metanalyst-Agent CLI Initializer"
echo "=================================="

# Verificar se Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado. Instale Python 3.8+ primeiro."
    exit 1
fi

# Verificar se pip está instalado
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 não encontrado. Instale pip primeiro."
    exit 1
fi

# Verificar e instalar dependências
echo "📦 Verificando dependências..."
if [ ! -f "requirements-cli.txt" ]; then
    echo "❌ requirements-cli.txt não encontrado!"
    exit 1
fi

# Instalar/atualizar dependências CLI
echo "⬇️ Instalando dependências do CLI..."
pip3 install -r requirements-cli.txt

if [ $? -ne 0 ]; then
    echo "❌ Erro ao instalar dependências!"
    exit 1
fi

# Verificar variáveis de ambiente essenciais
echo "🔑 Verificando variáveis de ambiente..."

missing_vars=()

if [ -z "$OPENAI_API_KEY" ]; then
    missing_vars+=("OPENAI_API_KEY")
fi

if [ -z "$TAVILY_API_KEY" ]; then
    missing_vars+=("TAVILY_API_KEY")
fi

if [ ${#missing_vars[@]} -gt 0 ]; then
    echo "⚠️ Variáveis de ambiente ausentes:"
    for var in "${missing_vars[@]}"; do
        echo "   - $var"
    done
    echo ""
    echo "📝 Configure essas variáveis antes de usar o sistema:"
    echo "   export OPENAI_API_KEY='sua_chave_openai'"
    echo "   export TAVILY_API_KEY='sua_chave_tavily'"
    echo ""
    echo "💡 Ou crie um arquivo .env na raiz do projeto com:"
    echo "   OPENAI_API_KEY=sua_chave_openai"
    echo "   TAVILY_API_KEY=sua_chave_tavily"
    echo ""
fi

# Verificar se arquivo .env existe
if [ -f ".env" ]; then
    echo "📄 Arquivo .env encontrado - carregando variáveis..."
    export $(grep -v '^#' .env | xargs)
fi

# Criar diretórios necessários
echo "📁 Criando diretórios necessários..."
mkdir -p data/{checkpoints,vector_store,backups}
mkdir -p logs

# Tornar script executável se necessário
chmod +x run_cli.py

echo ""
echo "✅ Inicialização concluída!"
echo ""
echo "🚀 Para executar o CLI:"
echo "   python3 run_cli.py"
echo ""
echo "💡 Ou com opções específicas:"
echo "   python3 run_cli.py --debug"
echo "   python3 run_cli.py --storage memory"
echo "   python3 run_cli.py --query 'sua consulta de meta-análise'"
echo ""

# Se argumentos foram passados, executar CLI diretamente
if [ $# -gt 0 ]; then
    echo "🔧 Executando CLI com argumentos: $@"
    python3 run_cli.py "$@"
else
    echo "👆 Execute um dos comandos acima para iniciar!"
fi
