import json

import pytest

from src.retriever import CardapioRetriever, hashing_embedder
from src.tools import Ferramentas


@pytest.fixture(scope="module")
def ferramentas():
    return Ferramentas(CardapioRetriever(hashing_embedder()))


def test_filtro_vegano_e_sem_gluten_e_respeitado(ferramentas):
    res = ferramentas.buscar_pratos("comida", tags=["vegano", "sem_gluten"], limite=10)["resultados"]
    assert res, "deveria encontrar ao menos um prato"
    for r in res:
        assert "vegano" in r["tags"] and "sem_gluten" in r["tags"]


def test_preco_maximo_e_respeitado(ferramentas):
    res = ferramentas.buscar_pratos("qualquer coisa", preco_max=20, limite=10)["resultados"]
    assert res and all(r["preco"] <= 20 for r in res)


def test_restaurante_fechado_nao_aparece(ferramentas):
    # r07 (Taquería) está fechado no dataset
    res = ferramentas.buscar_pratos("tacos mexicanos", limite=10)["resultados"]
    assert all(r["restaurante_id"] != "r07" for r in res)


def test_busca_traz_prato_relevante_primeiro(ferramentas):
    res = ferramentas.buscar_pratos("tapioca de carne de sol")["resultados"]
    assert res[0]["prato_id"] == "p16"


def test_tag_invalida_retorna_erro(ferramentas):
    assert "erro" in ferramentas.buscar_pratos("x", tags=["kosher"])


def test_calculo_de_entrega(ferramentas):
    # r05: taxa_base 1.99 + 1.50 * 0.8 km = 3.19 ; tempo 12 + 4 * 0.8 = 15.2 -> 15
    e = ferramentas.calcular_entrega("r05")
    assert e["taxa_entrega"] == pytest.approx(3.19)
    assert e["tempo_estimado_min"] == 15


def test_entrega_de_restaurante_fechado_retorna_erro(ferramentas):
    assert "erro" in ferramentas.calcular_entrega("r07")


def test_executar_lida_com_json_invalido(ferramentas):
    out = json.loads(ferramentas.executar("buscar_pratos", "{nao é json"))
    assert "erro" in out


def test_executar_ferramenta_desconhecida(ferramentas):
    out = json.loads(ferramentas.executar("pedir_pizza", "{}"))
    assert "erro" in out
