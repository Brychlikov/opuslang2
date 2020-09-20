from lark import Lark
from pathlib import Path


_current_path = Path(__file__).resolve()
_grammar_fname = _current_path.parent.joinpath("opuslang2.lark")

# parser = Lark.open(_grammar_fname, parser='lalr', debug=True)
parser = Lark.open(_grammar_fname, parser='earley')
