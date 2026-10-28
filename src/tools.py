"""Ferramentas que o LLM pode chamar (tool calling).

Regra de ouro: tudo que é número (preço, frete, tempo) vem daqui, nunca
do "chute" do modelo. O LLM só decide QUAL ferramenta chamar e como
explicar o resultado.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .retriever import CardapioRetriever

TAGS_VALIDAS = ["vegano", "vegetariano", "sem_gluten", "sem_lactose", "picante", "doce", "proteico", "serve_2"]

# Parâmetros da fórmula de entrega (fictícios, mas explícitos e testáveis).
CUSTO_POR_KM = 1.50
MINUTOS_POR_KM = 4


class Ferramentas:
    def __init__(self, retriever: CardapioRetriever):
        self.r = retriever

    # ---------- implementações ----------
    def buscar_pratos(self, consulta: str, preco_max: float | None = None,
                      tags: list[str] | None = None, limite: int = 5) -> dict:
        tags_invalidas = [t for t in (tags or []) if t not in TAGS_VALIDAS]
        if tags_invalidas:
            return {"erro": f"Tags inválidas: {tags_invalidas}. Use: {TAGS_VALIDAS}"}
        resultados = self.r.buscar(consulta, preco_max=preco_max, tags=tags, limite=min(limite, 10))
        return {
            "resultados": [
                {
                    "prato_id": res.prato["id"],
                    "prato": res.prato["nome"],
                    "descricao": res.prato["descricao"],
                    "preco": res.prato["preco"],
                    "tags": res.prato["tags"],
                    "restaurante_id": res.restaurante["id"],
                    "restaurante": res.restaurante["nome"],
                    "avaliacao": res.restaurante["avaliacao"],
                    "relevancia": round(res.score, 3),
                }
                for res in resultados
            ]
        }

    def calcular_entrega(self, restaurante_id: str) -> dict:
        rest = self.r.restaurantes.get(restaurante_id)
        if rest is None:
            return {"erro": f"Restaurante '{restaurante_id}' não encontrado."}
        if not rest["aberto"]:
            return {"erro": f"{rest['nome']} está fechado agora."}
        km = rest["distancia_km"]
        return {
            "restaurante": rest["nome"],
            "distancia_km": km,
            "taxa_entrega": round(rest["taxa_base"] + CUSTO_POR_KM * km, 2),
            "tempo_estimado_min": int(round(rest["tempo_preparo_min"] + MINUTOS_POR_KM * km)),
        }

    def detalhes_restaurante(self, restaurante_id: str) -> dict:
        rest = self.r.restaurantes.get(restaurante_id)
        if rest is None:
            return {"erro": f"Restaurante '{restaurante_id}' não encontrado."}
        pratos = [
            {"prato_id": p["id"], "prato": p["nome"], "preco": p["preco"], "tags": p["tags"]}
            for p in self.r.pratos if p["restaurante_id"] == restaurante_id
        ]
        return {**rest, "cardapio": pratos}

    # ---------- integração com o LLM ----------
    def registry(self) -> dict[str, Callable[..., dict]]:
        return {
            "buscar_pratos": self.buscar_pratos,
            "calcular_entrega": self.calcular_entrega,
            "detalhes_restaurante": self.detalhes_restaurante,
        }

    def executar(self, nome: str, argumentos_json: str) -> str:
        """Executa a ferramenta pedida pelo modelo, sempre devolvendo JSON.
        Erros viram mensagens para o modelo se corrigir, em vez de quebrar o app."""
        fn = self.registry().get(nome)
        if fn is None:
            return json.dumps({"erro": f"Ferramenta desconhecida: {nome}"}, ensure_ascii=False)
        try:
            args: dict[str, Any] = json.loads(argumentos_json or "{}")
            return json.dumps(fn(**args), ensure_ascii=False)
        except (json.JSONDecodeError, TypeError) as e:
            return json.dumps({"erro": f"Argumentos inválidos: {e}"}, ensure_ascii=False)


SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "buscar_pratos",
            "description": "Busca pratos no catálogo por significado (ex.: 'algo leve e proteico'), com filtros opcionais de preço e restrição alimentar. Retorna apenas restaurantes abertos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {"type": "string", "description": "O que a pessoa quer comer, em linguagem natural."},
                    "preco_max": {"type": "number", "description": "Preço máximo do prato em reais."},
                    "tags": {"type": "array", "items": {"type": "string", "enum": TAGS_VALIDAS},
                             "description": "Restrições obrigatórias. Todas precisam estar presentes no prato."},
                    "limite": {"type": "integer", "description": "Quantidade de resultados (padrão 5, máx. 10)."},
                },
                "required": ["consulta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_entrega",
            "description": "Calcula taxa de entrega e tempo estimado para um restaurante.",
            "parameters": {
                "type": "object",
                "properties": {"restaurante_id": {"type": "string", "description": "ID do restaurante, ex.: r02"}},
                "required": ["restaurante_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detalhes_restaurante",
            "description": "Mostra dados e o cardápio completo de um restaurante.",
            "parameters": {
                "type": "object",
                "properties": {"restaurante_id": {"type": "string", "description": "ID do restaurante, ex.: r05"}},
                "required": ["restaurante_id"],
            },
        },
    },
]
