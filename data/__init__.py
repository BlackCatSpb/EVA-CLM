"""Podgotovka dannyh dlya obucheniya WideBind.

Edinyy modul, ob"edinyayuschiy lingvisticheskiy frontend i binarizator:

- `sentence_builder` -- lingvisticheskiy postroitel predlozheniy s vesami
  vazhnosti po lambda_d (roli, padezhi, funkcionalnye slova);
- `build_streams_eos` -- CLI-binarizator: main-russian/{GENRE}/*.txt ->
  wb/token_stream_{GENRE}_eos.bin (uint16) s yavnymi granicami predlozheniy
  ([EOS] + predlozhenie + [EOS]), kotorye potreblyaet scripts/train.py.

Zapusk binarizacii:

    python -m data.build_streams_eos --src main-russian --out-dir wb
"""

from .sentence_builder import SentenceBuilder
from .build_streams_eos import main as build_streams_cli

__all__ = ["SentenceBuilder", "build_streams_cli"]