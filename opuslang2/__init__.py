from lark import Lark
from pathlib import Path

from opuslang2.parser import parser
from .compile import *


if __name__ == '__main__':
    test = """20-25, 4+ H"""

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

    tree = parser.parse(demo)
    # tree = parser.parse(test, start="test")
    print(tree.pretty())
