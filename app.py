"""Interface Gradio. Roda local (`python app.py`) ou no Hugging Face Spaces."""
from __future__ import annotations

import json
import os

import gradio as gr
from dotenv import load_dotenv

from src.agent import AgenteCardapio
from src.retriever import CardapioRetriever, hashing_embedder, sentence_transformer_embedder
from src.tools import Ferramentas

load_dotenv()

embedder = hashing_embedder() if os.getenv("EMBEDDER") == "hashing" else sentence_transformer_embedder()
agente = AgenteCardapio(Ferramentas(CardapioRetriever(embedder)))


def formatar_passos(passos) -> str:
    if not passos:
        return ""
    linhas = ["\n\n<details><summary>🔧 Ferramentas usadas</summary>\n"]
    for p in passos:
        args = json.loads(p.argumentos or "{}")
        linhas.append(f"- `{p.ferramenta}({json.dumps(args, ensure_ascii=False)})`")
    linhas.append("\n</details>")
    return "\n".join(linhas)


def chat(mensagem: str, historico: list[dict]) -> str:
    try:
        resp = agente.responder(mensagem, historico)
    except Exception as e:  # erro de API, chave ausente etc.
        return f"⚠️ Erro ao chamar o modelo: {e}"
    return resp.texto + formatar_passos(resp.passos)


demo = gr.ChatInterface(
    fn=chat,
    type="messages",
    title="🍽️ Assistente de Cardápio",
    description="Agente com RAG + tool calling que recomenda pratos por gosto, restrição e orçamento. "
                "Dados fictícios, para demonstração.",
    examples=[
        "Quero algo vegano e sem glúten até R$ 40",
        "Tenho R$ 30 no total, com entrega. O que chega mais rápido?",
        "Me indica um doce para sobremesa, sem lactose",
        "Comida japonesa sem nada picante",
    ],
)

if __name__ == "__main__":
    demo.launch()
