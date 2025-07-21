# Correção de UUIDs Inválidos - Metanalyst-Agent

## 📋 Problema Identificado

O sistema estava apresentando erros de sintaxe UUID no PostgreSQL:

```
Database connection error: invalid input syntax for type uuid: "AF_amiodarone_beta"
```

### 🔍 Causa Raiz

O problema ocorreu porque em alguma execução anterior, foi gerado um `meta_analysis_id` baseado no conteúdo PICO (AF = Atrial Fibrillation, amiodarone = intervenção, beta = beta blockers) em vez de um UUID válido no formato padrão.

### 📊 Impacto

- Falha na conexão com banco de dados
- Impossibilidade de executar meta-análises
- Corrupção de dados relacionais

## 🛠️ Soluções Implementadas

### 1. Scripts de Limpeza

#### `scripts/cleanup_invalid_uuids.py`
- Script Python interativo para identificar e remover UUIDs inválidos
- Faz backup automático antes da remoção
- Adiciona constraints de validação para prevenir problemas futuros

#### `scripts/fix_invalid_uuids.sql`
- Script SQL direto para limpeza manual
- Remove registros com UUIDs no formato incorreto
- Reindexação automática das tabelas

#### `scripts/fix_uuids.sh`
- Script bash executável para automação
- Verifica pré-requisitos (DATABASE_URL, diretório, etc.)
- Executa limpeza com feedback visual

### 2. Validação Preventiva

#### Funções de Validação em `research_tools.py`

```python
def validate_uuid_format(uuid_string: str) -> bool:
    """Valida se uma string tem formato de UUID válido."""
    
def ensure_valid_uuid(meta_analysis_id: str, operation: str = "operation") -> str:
    """Garante que um meta_analysis_id seja um UUID válido."""
```

#### Implementação nos Tools

- `search_literature()`: Valida meta_analysis_id antes de usar
- Geração automática de UUID válido se detectado formato incorreto
- Logging detalhado para debugging

### 3. Constraints de Banco

Adicionamos constraints PostgreSQL para validação automática:

```sql
ALTER TABLE articles 
ADD CONSTRAINT valid_meta_analysis_id_format 
CHECK (meta_analysis_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$');
```

## 🚀 Como Executar a Correção

### Opção 1: Script Automático (Recomendado)

```bash
# Tornar executável (já feito)
chmod +x scripts/fix_uuids.sh

# Executar correção
./scripts/fix_uuids.sh
```

### Opção 2: Script Python Direto

```bash
python3 scripts/cleanup_invalid_uuids.py
```

### Opção 3: SQL Manual

```bash
psql $DATABASE_URL -f scripts/fix_invalid_uuids.sql
```

## ✅ Verificação da Correção

### Teste de Conectividade

```bash
python3 -c "
from metanalyst_agent.database.connection import get_database_manager
db = get_database_manager()
print('✅ Conexão OK') if db.health_check()['status'] == 'healthy' else print('❌ Falha')
"
```

### Teste de UUIDs

```bash
python3 test_critical_fixes.py
```

Resultado esperado: **5/5 testes passaram**

### Verificação Manual no Banco

```sql
-- Verificar se ainda há UUIDs inválidos
SELECT COUNT(*) FROM articles 
WHERE meta_analysis_id !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$';

-- Deve retornar 0
```

## 🔒 Prevenção Futura

### 1. Validação Automática
- Todas as ferramentas agora validam UUIDs antes de usar
- Geração automática de novos UUIDs válidos quando necessário

### 2. Constraints de Banco
- PostgreSQL rejeita automaticamente UUIDs inválidos
- Proteção em nível de banco de dados

### 3. Logging Detalhado
- Warnings automáticos quando UUIDs inválidos são detectados
- Tracking da origem dos problemas

### 4. Testes Automatizados
- Script `test_critical_fixes.py` verifica integridade do sistema
- Pode ser executado regularmente para monitoramento

## 📊 Resultados dos Testes

Após implementação das correções:

```
🎯 Resultado: 5/5 testes passaram
✅ TODAS AS CORREÇÕES FUNCIONANDO!

📊 RESUMO DOS TESTES:
   Conexão com banco: ✅ PASSOU
   StateManager: ✅ PASSOU  
   Persistência de URLs: ✅ PASSOU
   Busca de literatura: ✅ PASSOU
   End-to-end: ✅ PASSOU
```

## 🔍 Monitoramento Contínuo

### Logs a Observar

```bash
# Verificar warnings de UUID inválido
grep "Invalid UUID format" logs/cli.log

# Verificar erros de banco relacionados
grep "Database connection error" logs/cli.log
```

### Saúde do Sistema

Execute periodicamente:

```bash
python3 test_critical_fixes.py
```

## 📞 Suporte

Se o problema persistir:

1. ✅ Verifique se DATABASE_URL está corretamente configurada
2. ✅ Execute os scripts de limpeza
3. ✅ Verifique os logs em `logs/cli.log`
4. ✅ Execute os testes de verificação
5. ✅ Confirme que as constraints foram adicionadas corretamente

---

**Status:** ✅ **PROBLEMA RESOLVIDO**
**Data:** 20/01/2025
**Versão:** v1.0.1-uuid-fix
