# 🔬 Metanalyst-Agent CLI

Interface de linha de comando robusta para o sistema automatizado de meta-análise com logs de debug detalhados.

## 🚀 Inicialização Rápida

### 1. Setup Automático
```bash
# Torna os scripts executáveis e inicializa o ambiente
chmod +x start_cli.sh run_cli.py
./start_cli.sh
```

### 2. Execução Manual
```bash
# Instalar dependências
pip install -r requirements-cli.txt

# Executar CLI
python3 run_cli.py
```

## 🎯 Modos de Uso

### Modo Interativo (Recomendado)
```bash
python3 run_cli.py
```
- Interface rica com menus
- Progress bars em tempo real
- Múltiplos modos de debug
- Histórico de análises
- Configurações dinâmicas

### Modo Direto (Command Line)
```bash
# Executar meta-análise diretamente
python3 run_cli.py --query "eficácia da meditação mindfulness para ansiedade"

# Com opções específicas
python3 run_cli.py \
  --query "mindfulness vs CBT para ansiedade" \
  --max-articles 30 \
  --quality-threshold 0.85 \
  --max-time 45 \
  --stream-mode debug \
  --debug
```

## 🔧 Opções de Linha de Comando

| Opção | Descrição | Padrão |
|-------|-----------|---------|
| `--debug` / `-d` | Habilita modo debug detalhado | `False` |
| `--storage` / `-s` | Tipo de storage (`memory`, `postgres`) | `postgres` |
| `--query` / `-q` | Query de meta-análise direta | - |
| `--max-articles` / `-m` | Máximo de artigos para processar | `50` |
| `--quality-threshold` / `-t` | Threshold de qualidade (0.0-1.0) | `0.8` |
| `--max-time` | Tempo máximo em minutos | `30` |
| `--stream-mode` | Modo de debug (`values`, `updates`, `debug`) | `updates` |

## 🎨 Interface Rica

### Banner Inicial
```
╭─────────────────────────────────────────────────────────────────╮
│                     🔬 METANALYST-AGENT CLI                     │
│                                                                 │
│              Sistema Automatizado de Meta-Análise              │
│                     com Agentes Multi-LLM                      │
╰─────────────────────────────────────────────────────────────────╯

🗄️ Storage: PostgreSQL
🧠 Model: gpt-4.1
🐛 Debug: OFF
📁 Logs: logs/
```

### Menu Principal
```
Opções disponíveis:
1. 🔬 Nova Meta-Análise
2. 📊 Listar Análises Anteriores
3. ⚙️ Configurações
4. 🔍 Debug/Teste
5. ❓ Ajuda
6. 🚪 Sair
```

## 🐛 Modos de Debug Avançados

### 1. Modo `values` (Estado Completo)
```bash
python3 run_cli.py --stream-mode values --debug
```
- Estado completo a cada iteração
- Histórico de mensagens completo
- Estatísticas em tempo real

### 2. Modo `updates` (Por Agente)
```bash
python3 run_cli.py --stream-mode updates
```
- Updates específicos por agente
- Fluxo de handoffs entre agentes
- Progresso detalhado

### 3. Modo `debug` (Máximo Detalhamento)
```bash
python3 run_cli.py --stream-mode debug --debug
```
- Estado interno completo
- Decisões de roteamento
- Payload de cada nó
- Logs de sistema detalhados

## 📊 Tracking de Progresso

### Progress Bar em Tempo Real
```
⣾ Fase: extraction | Encontrados: 25 | Processados: 8  [00:02:34]
```

### Tabela de Resumo
```
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Metric          ┃ Value  ┃ Status ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ Elapsed Time    │ 45.2s  │ ⏱️      │
│ Current Phase   │ Search │ 📍     │
│ Articles Found  │ 25     │ 🔍     │
│ Articles Processed │ 8   │ ✅     │
│ Avg Quality Score │ 0.82 │ 📊     │
│ Most Active Agent │ Processor │ 🤖 │
└─────────────────┴────────┴────────┘
```

## 🎯 Formatação de Output

### Mensagens de Agentes
```
[08:34:12] 🔄 Node 'researcher'
   📨 Messages: 3 total
   └─ Last: 🤖 Researcher: Encontrei 25 artigos relevantes sobre...
   🔍 Articles Found: 25
   📍 Phase: search
```

### Erros Formatados
```
[08:35:02] ❌ Error in meta-analysis execution
   🏷️ Type: ConnectionError
   💬 Message: Failed to connect to API endpoint
```

## 🔍 Funcionalidades de Debug

### 1. Teste de Conexão com Agente
- Inicialização de agentes
- Geração de PICO framework
- Verificação de models

### 2. Teste de Busca Tavily
- Conectividade com API
- Resultados de busca
- Qualidade dos dados

### 3. Verificação de Sistema
- Status de diretórios
- Variáveis de ambiente
- Imports de bibliotecas

### 4. Teste de Banco de Dados
- Conectividade PostgreSQL
- Operações de checkpointer
- Operações de store

### 5. Logs de Teste
- Geração de logs em diferentes níveis
- Verificação de arquivos
- Sistema de logging

## 📝 Logs Detalhados

### Arquivo de Log Automático
```
logs/cli.log
```

### Formato de Log
```
2025-07-20 08:34:12,345 - metanalyst_cli - INFO - Meta-analysis started - ID: abc123
2025-07-20 08:34:15,678 - metanalyst_cli - DEBUG - Full state for node researcher: {...}
2025-07-20 08:34:18,901 - metanalyst_cli - ERROR - Processor failed: Connection timeout
```

## 💾 Salvamento de Resultados

### Formato JSON Estruturado
```json
{
  "meta_analysis_id": "abc123",
  "timestamp": "20250720_083412",
  "pico": {
    "P": "Adultos com ansiedade",
    "I": "Meditação mindfulness",
    "C": "Terapia cognitivo-comportamental",
    "O": "Redução dos sintomas de ansiedade"
  },
  "statistics": {
    "articles_found": 25,
    "articles_processed": 18,
    "articles_failed": 3
  },
  "quality_scores": {
    "researcher": 0.85,
    "processor": 0.78,
    "analyst": 0.92
  },
  "final_report": "...",
  "citations": [...]
}
```

## 🔧 Configurações Dinâmicas

### Modificar em Tempo de Execução
- Toggle Debug Mode
- Toggle Storage Type  
- Change Recursion Limit
- Ajustar thresholds

### Exemplo de Modificação
```
🔧 Modificar Configurações

1. Toggle Debug Mode
2. Toggle Storage Type
3. Change Recursion Limit
4. Voltar

Escolha: 1
Debug mode: ON
```

## 📊 Histórico de Análises

### Tabela de Análises Anteriores
```
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Data               ┃ ID       ┃ Pergunta                ┃ Artigos  ┃ Status        ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ 20250720_083412    │ abc123   │ mindfulness para ansied...│ 18       │ ✅ Completa   │
│ 20250720_073505    │ def456   │ CBT vs medicação depres...│ 12       │ 📝 Rascunho   │
└────────────────────┴──────────┴─────────────────────────┴──────────┴───────────────┘
```

## 🚦 Variáveis de Ambiente

### Obrigatórias
```bash
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."
```

### Opcionais
```bash
export DATABASE_URL="postgresql://user:pass@localhost:5432/metanalysis"
```

### Arquivo `.env`
```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
DATABASE_URL=postgresql://user:pass@localhost:5432/metanalysis
```

## 🛠️ Troubleshooting

### Problemas Comuns

#### 1. Erro de API Keys
```bash
❌ Missing: OPENAI_API_KEY
💡 Configure: export OPENAI_API_KEY='sua_chave'
```

#### 2. Erro de Conexão de Banco
```bash
❌ DATABASE_URL não configurado
💡 Configure PostgreSQL ou use --storage memory
```

#### 3. Timeout de Execução
```bash
⏰ Meta-análise interrompida por timeout após 30.0min
💡 Aumente --max-time ou reduza --max-articles
```

#### 4. Limite de Recursão
```bash
🔄 Limite de recursão atingido: Recursion limit exceeded
💡 Use modo debug para identificar loops
```

### Logs para Debug
```bash
# Verificar logs
tail -f logs/cli.log

# Executar com debug máximo
python3 run_cli.py --debug --stream-mode debug
```

## 📚 Ajuda Contextual

### Comando Help Integrado
```
❓ Ajuda do Metanalyst-Agent CLI

## 🔬 Sobre o Sistema
O Metanalyst-Agent é um sistema automatizado...

## 🚀 Funcionalidades Principais
1. Meta-Análise Automatizada
2. Sistema Multi-Agente
3. Modos de Debug

## 📊 Framework PICO
- Population: População estudada
- Intervention: Intervenção/exposição
- Comparison: Comparação/controle
- Outcome: Desfecho/resultado
```

## 🎉 Exemplos Práticos

### Exemplo 1: Análise Simples
```bash
python3 run_cli.py --query "meditação mindfulness para depressão"
```

### Exemplo 2: Análise com Debug
```bash
python3 run_cli.py \
  --query "CBT vs SSRI para transtorno de ansiedade" \
  --debug \
  --stream-mode debug \
  --max-articles 20
```

### Exemplo 3: Análise Rápida (In-Memory)
```bash
python3 run_cli.py \
  --storage memory \
  --query "exercício físico para fibromialgia" \
  --max-time 15
```

---

## 🤝 Contribuindo

Para contribuir com melhorias no CLI:

1. Fork o repositório
2. Crie sua feature branch
3. Implemente melhorias
4. Teste extensivamente 
5. Submit pull request

## 📞 Suporte

Para dúvidas e problemas:
- Verifique logs em `logs/cli.log`
- Use modo debug para diagnosticar
- Consulte troubleshooting acima
- Abra issue no GitHub se necessário

---

**Metanalyst-Agent CLI** - Sistema Automatizado de Meta-Análise com Interface Rica e Debug Avançado 🔬
