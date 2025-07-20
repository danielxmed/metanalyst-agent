# LangGraph e Fluxos

LangGraph permite compor agentes LLM em um grafo de estados. Cada nó pode atualizar o estado compartilhado, que é persistido em banco. Utilizamos checkpointers para memória de curto prazo e stores para memória de longo prazo. Veja `CLAUDE.md` para mais detalhes de implementação.
