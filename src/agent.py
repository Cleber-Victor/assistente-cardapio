"""Loop do agente: LLM decide ferramentas -> executamos -> LLM responde.

Usa a API compatível com OpenAI, então funciona com Groq (gratuito),
OpenAI, OpenRouter ou um servidor local (Ollama), só trocando variáveis
de ambiente.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .tools import SCHEMAS, Ferramentas

SYSTEM_PROMPT = """Você é o Assistente de Cardápio, um ajudante de pedidos de delivery.

Regras:
- Use SEMPRE as ferramentas para buscar pratos, preços, frete e tempo. Nunca invente pratos, preços ou restaurantes.
- Se a pessoa mencionar restrição (vegano, sem glúten, sem lactose...), passe-a no filtro `tags`.
- Se mencionar orçamento, use `preco_max`. Se o orçamento incluir entrega, calcule o frete antes de recomendar.
- Recomende no máximo 3 opções, dizendo prato, restaurante, preço e por que combina com o pedido.
- Se nada atender, diga isso claramente e sugira relaxar algum critério.
- Responda em português do Brasil, de forma curta e simpática.
"""

MAX_PASSOS = 6


@dataclass
class Passo:
    ferramenta: str
    argumentos: str
    resultado: str


@dataclass
class Resposta:
    texto: str
    passos: list[Passo] = field(default_factory=list)


class AgenteCardapio:
    def __init__(self, ferramentas: Ferramentas, client: Any | None = None, model: str | None = None):
        self.ferramentas = ferramentas
        if client is None:
            from openai import OpenAI  # import tardio: testes usam cliente falso
        self.client = client or OpenAI(
            base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY"),
        )
        self.model = model or os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

    def responder(self, mensagem: str, historico: list[dict] | None = None) -> Resposta:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        msgs += [{"role": m["role"], "content": m["content"]} for m in (historico or [])
                 if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
        msgs.append({"role": "user", "content": mensagem})

        passos: list[Passo] = []
        for _ in range(MAX_PASSOS):
            resp = self.client.chat.completions.create(
                model=self.model, messages=msgs, tools=SCHEMAS, tool_choice="auto", temperature=0.2,
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return Resposta(msg.content or "", passos)

            msgs.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                resultado = self.ferramentas.executar(tc.function.name, tc.function.arguments)
                passos.append(Passo(tc.function.name, tc.function.arguments, resultado))
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": resultado})

        return Resposta("Desculpe, não consegui concluir a busca. Pode reformular o pedido?", passos)
