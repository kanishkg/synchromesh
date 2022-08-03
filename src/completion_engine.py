from lark import Lark, Token
from lark.exceptions import UnexpectedCharacters

import regex


class CompletionEngine:
    def complete(self, prefix: str) -> regex.Pattern:
        raise NotImplementedError()


class LarkCompletionEngine(CompletionEngine):
    def __init__(self, grammar, start_token):
        self.parser = Lark(grammar, start=start_token, parser='lalr', regex=True)
        self.terminal_dict = self.parser._terminals_dict
    
    def complete(self, prefix: str) -> regex.Pattern:
        interactive_parser = self.parser.parse_interactive(prefix)
        token = None
        try:
            for token in interactive_parser.parser_state.lexer.lex(interactive_parser.parser_state): 
                interactive_parser.parser_state.feed_token(token)
        except UnexpectedCharacters as e:
            print("Unexpected character, string is not complete")
        valid_tokens = interactive_parser.accepts()
        # get the regex for the valid tokens
        valid_regex = [fr'{self.terminal_dict[t].pattern}' for t in valid_tokens if t!='$END']
        return valid_regex


def main():
    json_grammar = r"""
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

        """
    json_comp_engine = LarkCompletionEngine(json_grammar, 'value')
    text = '{"a": 1, "b": 2, "c": {"d": 3, "e":'
    valid_regexes = json_comp_engine.complete(text)
    print(valid_regexes)
    # end_token = Token.new_borrow_pos('$END', '', token) if token else Token('$END', '', 0, 1, 1)
    # interactive_parser.parser_state.feed_token(end_token, True)

    
if __name__ == '__main__':
    main()
