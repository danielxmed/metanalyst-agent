# Arquitetura

O sistema usa LangGraph para controlar o fluxo entre os agentes. O orquestrador decide qual agente deve ser executado de acordo com o estado atual. Toda a memória de curto e longo prazo é salva no PostgreSQL por meio dos componentes `PostgresSaver` e `PostgresStore`.
