from lark import Lark
from pathlib import Path


_current_path = Path(__file__).resolve()
_grammar_fname = _current_path.parent.joinpath("opuslang2.lark")

# parser = Lark.open(_grammar_fname, parser='lalr', debug=True)
parser = Lark.open(_grammar_fname, parser='earley', debug=True)

test = """
open {
    1C {
        13-14, 4+ C
        15-16, 2+H
    }
}
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

# demo = open("../opuslang2-2.demo").read()

tree = parser.parse(test_test)
# tree = parser.parse(test_test)
print(tree.pretty())
