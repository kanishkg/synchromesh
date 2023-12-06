from transformers import AutoTokenizer, AutoModelForCausalLM

from synchromesh import LarkCompletionEngine, HuggingFaceModel, predict_constrained

def test_college_grammar_csd():
    """This is a simple example of using a grammar for semantic parsing.

    Suppose the assistant's role is to take a natural language question and rewrite it
    as a string accepted by a background grammar. CSD will ensure that the language model's
    output will always parse, even though the model itself is not aware of the grammar."""

    college_grammar = r"""
        ?request: function " of " dept code
        function: "instructor" | "students" | "capacity" | "deptcode" | "school" | "college"
        dept:  /[A-Z]{3}/
        code: /[0-9]{3}/
    """

    college_prompt = """Paraphrase the following sentences
Human: who teaches CSE101?
Assistant:instructor of CSE101
Human: how many students can enroll in PSY456?
Assistant:capacity of PSY456
Human: what's the department of BIO433?
Assistant:"""

    num_samples = 10
    comp_engine = LarkCompletionEngine(college_grammar, 'request', False)

    # Can be any huggingface model string or local path to weights.
    HF_MODEL = 'gpt2'
    gpt2 = AutoModelForCausalLM.from_pretrained(HF_MODEL, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

    # These should work too:
    # m = RandomLanguageModel()
    # m = OpenAIModel(model="text-ada-001", prompt_template=college_prompt, temperature=1.)
    # Note that OpenAI now considers the Completions API as "legacy", which we use for their models.

    lm = HuggingFaceModel(gpt2, tokenizer=tokenizer, prompt_template=college_prompt, temperature=0.25)

    for i in range(num_samples):
        print(HF_MODEL, "prediction:", predict_constrained(comp_engine, lm , 1, True, stop_tokens=["\n"]))


def test_fast_forward():
    # This test shows the "Fast-forwarding" behavior of sampling with an OpenAI model:
    # this implementation of CSD will keep going until there is a choice to be made. Here, since
    # the grammar will only have a single option for the answer, the model will not be repeatedly
    # queried for each mistake it makes. If there were two options, fast-forwarding would
    # first fast-forward to

    fixed_response = r"""
        ?response: "the answer is abcdefghijklmnopqrstuvwxyz"
    """

    prompt = """You are a helpful assistant.

Human: What day is today?
Assistant: Thursday

Human: What is the answer?
Assistant:"""

    num_samples = 1
    api_key = os.environ.get('OPENAI_API_KEY')
    for i in range(num_samples):
        comp_engine = LarkCompletionEngine(fixed_response, 'response', False)

    ada = OpenAIModel(model="text-ada-001", prompt_template=prompt,
                      api_key=api_key, temperature=1.)
    print(predict_constrained(comp_engine, ada, 1, True,
                              stop_tokens=["\n"], fast_forward=True))



if __name__ == '__main__':
    test_college_grammar_csd()
