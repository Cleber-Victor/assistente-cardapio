"""Busca semântica de pratos com embeddings + filtros estruturados.

Design: o "embedder" é injetável. Em produção usamos sentence-transformers
(modelo multilíngue que entende português); nos testes usamos um embedder
falso e determinístico, para rodar sem baixar modelo nem acessar a internet.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cardapio.json"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    def __call__(self, textos: Sequence[str]) -> np.ndarray: ...


def sentence_transformer_embedder(model_name: str = DEFAULT_MODEL) -> Embedder:
    """Carrega o modelo só quando necessário (import pesado)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(textos: Sequence[str]) -> np.ndarray:
        return model.encode(list(textos), normalize_embeddings=True)

    return embed


@dataclass
class Resultado:
    prato: dict
    restaurante: dict
    score: float


class CardapioRetriever:
    def __init__(self, embedder: Embedder, data_path: Path = DATA_PATH):
        dados = json.loads(Path(data_path).read_text(encoding="utf-8"))
        self.restaurantes = {r["id"]: r for r in dados["restaurantes"]}
        self.pratos = dados["pratos"]
        self.embedder = embedder
        # Texto indexado: nome + descrição + categoria + tags, para a busca
        # entender pedidos como "algo japonês" ou "comida sem leite".
        textos = [self._texto_indexado(p) for p in self.pratos]
        self.matriz = self._normalizar(np.asarray(embedder(textos), dtype=np.float32))

    def _texto_indexado(self, prato: dict) -> str:
        rest = self.restaurantes[prato["restaurante_id"]]
        tags = ", ".join(t.replace("_", " ") for t in prato["tags"])
        return f"{prato['nome']}. {prato['descricao']} Culinária {rest['categoria']}. {tags}"

    @staticmethod
    def _normalizar(m: np.ndarray) -> np.ndarray:
        normas = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.clip(normas, 1e-12, None)

    def buscar(
        self,
        consulta: str,
        preco_max: float | None = None,
        tags: list[str] | None = None,
        apenas_abertos: bool = True,
        limite: int = 5,
    ) -> list[Resultado]:
        """Filtra primeiro (regras exatas), depois ordena por similaridade.

        Filtros duros evitam que o modelo recomende um prato com carne para
        quem pediu vegano só porque a descrição "parecia" com a consulta.
        """
        q = self._normalizar(np.asarray(self.embedder([consulta]), dtype=np.float32))[0]
        scores = self.matriz @ q
        tags = [t.strip().lower() for t in (tags or []) if t.strip()]

        resultados = []
        for prato, score in zip(self.pratos, scores):
            rest = self.restaurantes[prato["restaurante_id"]]
            if apenas_abertos and not rest["aberto"]:
                continue
            if preco_max is not None and prato["preco"] > preco_max:
                continue
            if any(t not in prato["tags"] for t in tags):
                continue
            resultados.append(Resultado(prato, rest, float(score)))

        resultados.sort(key=lambda r: r.score, reverse=True)
        return resultados[: max(1, limite)]


def hashing_embedder(dim: int = 256) -> Callable[[Sequence[str]], np.ndarray]:
    """Embedder leve por "bag of words" com hashing. Usado nos testes e como
    fallback offline. Não entende sinônimos, mas é determinístico."""
    import hashlib
    import re
    import unicodedata

    def tokens(texto: str) -> list[str]:
        t = unicodedata.normalize("NFKD", texto.lower())
        t = "".join(c for c in t if not unicodedata.combining(c))
        return re.findall(r"[a-z0-9]+", t)

    def embed(textos: Sequence[str]) -> np.ndarray:
        m = np.zeros((len(textos), dim), dtype=np.float32)
        for i, texto in enumerate(textos):
            for tok in tokens(texto):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                m[i, h % dim] += 1.0
        return m

    return embed
