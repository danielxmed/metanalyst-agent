#!/bin/bash

# Script para corrigir UUIDs inválidos no banco de dados
# Este script executa a limpeza automática dos registros com meta_analysis_id inválido

echo "🔧 Corrigindo UUIDs inválidos no banco de dados do Metanalyst-Agent..."
echo ""

# Verificar se estamos no diretório correto
if [ ! -f "metanalyst_agent/__init__.py" ]; then
    echo "❌ Execute este script do diretório raiz do projeto metanalyst-agent"
    exit 1
fi

# Verificar se DATABASE_URL está configurado
if [ -z "$DATABASE_URL" ]; then
    echo "❌ DATABASE_URL não está configurado!"
    echo "   Configure com: export DATABASE_URL='sua_url_do_postgresql'"
    exit 1
fi

echo "✅ DATABASE_URL configurado"
echo "📍 Banco: $(echo $DATABASE_URL | sed 's/.*@\([^/]*\).*/\1/' | cut -d: -f1)"
echo ""

# Executar o script Python de limpeza
echo "🐍 Executando script de limpeza..."
python3 scripts/cleanup_invalid_uuids.py

# Verificar se o script foi bem-sucedido
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Correção de UUIDs concluída com sucesso!"
    echo ""
    echo "🔍 Próximos passos recomendados:"
    echo "   1. Execute uma nova meta-análise para testar"
    echo "   2. Monitore os logs para verificar se não há mais erros de UUID"
    echo "   3. Os constraints adicionados previnem futuros problemas"
    echo ""
else
    echo ""
    echo "❌ Falha na correção. Verifique os logs acima para mais detalhes."
    echo ""
    echo "🔧 Soluções alternativas:"
    echo "   1. Execute manualmente: psql \$DATABASE_URL -f scripts/fix_invalid_uuids.sql"
    echo "   2. Verifique se o banco PostgreSQL está acessível"
    echo "   3. Confirme se as tabelas existem no banco"
    echo ""
    exit 1
fi
