from lark import Lark
from pathlib import Path


_current_path = Path(__file__).resolve()
_grammar_fname = _current_path.parent.joinpath("opuslang2.lark")

# parser = Lark.open(_grammar_fname, parser='lalr', debug=True)
parser = Lark.open(_grammar_fname, parser='earley', debug=True)

test = """$balance >= 1
"""

test_test = """

open {
    1C {
        13-14, 4+ C; [1]
        # 25+, 4+ H|S
    }
    1D {
        15 - 32, 5= H&S
    }
}

1C {
    1NT {
        15+, $balance >= 1
        $balance >= 1 and (H >= 3 or C|D == 5)
    }
}
"""

demo = open("../blas.ol2").read()

# tree = parser.parse(demo)
tree = parser.parse(demo)
print(tree.pretty())
