from lark import Lark, Token
from lark.exceptions import UnexpectedCharacters

import regex


class CompletionEngine:
    def complete(self, prefix: str) -> regex.Pattern:
        raise NotImplementedError()


class LarkCompletionEngine(CompletionEngine):
    def __init__(self, grammar):
        pass


# TODO: make this a class
def get_completions(text, parser):
    interactive_parser = parser.parse_interactive(text)
    interactive_parser = interactive_parser.as_immutable()
    token = None
    try:
        for token in interactive_parser.parser_state.lexer.lex(interactive_parser.parser_state): 
            interactive_parser.parser_state.feed_token(token)
    except UnexpectedCharacters as e:
        print("Unexpected character, string is not complete")
    return interactive_parser.accepts()

def main():
    json_parser = Lark(r"""
        ?value: dict
            | list
            | string
            | SIGNED_NUMBER      -> number
            | "true"             -> true
            | "false"            -> false
            | "null"             -> null

        list : "[" [value ("," value)*] "]"

        dict : "{" [pair ("," pair)*] "}"
        pair : string ":" value

        string : ESCAPED_STRING

        %import common.ESCAPED_STRING
        %import common.SIGNED_NUMBER
        %import common.WS
        %ignore WS

        """, start='value', parser='lalr')

    text = '{"a": 1, "b": 2, "c": {"d": 3, "e": 4}}'
    for l in range(len(text)):
        print(f"parsing {text[:l]}")
        print(get_completions(text[:l], json_parser))
    # end_token = Token.new_borrow_pos('$END', '', token) if token else Token('$END', '', 0, 1, 1)
    # interactive_parser.parser_state.feed_token(end_token, True)

    
if __name__ == '__main__':
    main()
