"""Testa o loop do agente com um cliente LLM falso (sem internet, sem custo)."""
from types import SimpleNamespace as NS

from src.agent import AgenteCardapio
from src.retriever import CardapioRetriever, hashing_embedder
from src.tools import Ferramentas


class ClienteFalso:
    """Na 1ª chamada pede uma ferramenta; na 2ª responde com texto."""

    def __init__(self):
        self.chamadas = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.chamadas.append(kwargs["messages"])
        if len(self.chamadas) == 1:
            tc = NS(id="call_1", function=NS(name="calcular_entrega", arguments='{"restaurante_id": "r05"}'))
            msg = NS(content=None, tool_calls=[tc])
        else:
            msg = NS(content="A entrega custa R$ 3,19.", tool_calls=None)
        return NS(choices=[NS(message=msg)])


def test_loop_executa_ferramenta_e_devolve_resultado_ao_modelo():
    cliente = ClienteFalso()
    agente = AgenteCardapio(Ferramentas(CardapioRetriever(hashing_embedder())), client=cliente, model="fake")

    resp = agente.responder("Quanto é a entrega da tapiocaria?")

    assert resp.texto == "A entrega custa R$ 3,19."
    assert [p.ferramenta for p in resp.passos] == ["calcular_entrega"]
    # a 2ª chamada ao modelo precisa conter o resultado real da ferramenta
    msg_tool = cliente.chamadas[1][-1]
    assert msg_tool["role"] == "tool" and "3.19" in msg_tool["content"]
