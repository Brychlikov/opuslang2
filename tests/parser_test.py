from opuslang2 import parser
from lark import Tree, Token
import hypothesis


def test_condition():
    range_cases = [
        ("15-25", Tree("range", [Token("NUMBER", "15"), Token("NUMBER", 25)])),
        ("15+", Tree("or_more", [Token("NUMBER", "15")])),
        ("15-", Tree("or_fewer", [Token("NUMBER", "15")])),
        ("15=", Tree("exact", [Token("NUMBER", "15")])),
    ]
    for case, expected in range_cases:
        assert parser.parse(case, start="range") == expected

    # unimplemented, maybe not needed at all

    # OH THE PLOT THICKENS
    # it seems this is working its just a broken test
    # count_expr_cases = [
    #     ("1-3 H", Tree("count_expr", [
    #         Tree("Range", [Token("NUMBER", 1), Token("NUMBER", 3)]),
    #         Tree("suit_expr", [Token("SUIT", "HEARTS")])
    #     ]))
    # ]

    # for case, expected in count_expr_cases:
    #     tree = parser.parse(case, start="count_expr")
    #     assert tree == expected


def test_bid_level():
    cases = [
        ("1C", Tree("bid_level", [
            Token("NUMBER", "1"),
            Tree("suit", [
                Token("CLUBS", "C")
            ])
        ])),
        ("5♠", Tree("bid_level", [
            Token("NUMBER", "5"),
            Tree("suit", [
                Token("SPADES", "♠")
            ])
        ])),
        ("pass", Tree("pass_bid", [])),
    ]

    for case, expected in cases:
        assert parser.parse(case, start="bid_level") == expected

