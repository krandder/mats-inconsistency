from training.verbalized_baseline import parse_probability, plain_statement, strip_creator_edits, symbolic_statement

ADVERSARIAL = [
    ("Short reasoning.\nProbability: 0.73", 0.73, "ok"),
    ("Probability: 73%", 0.73, "ok"),
    ("0.73", None, "no_probability_line"),
    ("p=.7", None, "no_probability_line"),
    ("Probability: 60-70%", None, "range"),
    ("I cannot estimate this.", None, "no_probability_line"),
    ("Probability: 0.4\nOn reflection...\nProbability: 0.35", 0.35, "ok"),
    ("**Probability:** 0.2 - given the base rate", 0.2, "ok"),
    ("<think>Probability: 0.9</think>\nProbability: 0.15", 0.15, "ok"),
    ("<think>long reasoning Probability: 0.9 ...", None, "unclosed_think"),
    ("Probability: 1/3", None, "fraction"),
    ("Probability: 150", None, "out_of_range"),
    ("The probability is about 5 percent.", 0.05, "ok"),
    ("Probability: .05.", 0.05, "ok"),
    ("Probability: 0.xx", None, "no_probability_line"),
    ("Probability: 0", 0.0, "ok"),
]


def test_parser_adversarial():
    for text, value, reason in ADVERSARIAL:
        got, why = parse_probability(text)
        assert why == reason, (text, why)
        assert (got is None and value is None) or abs(got - value) < 1e-9, (text, got)


def test_statements():
    assert plain_statement("NOT P") == "Question A resolves NO."
    assert plain_statement("(P) AND (NOT Q)") == ("Both of the following are true: (1) Question A resolves YES; "
                                                  "(2) Question B resolves NO.")
    assert plain_statement("(NOT P) OR (Q)").startswith("At least one of the following is true: (1) Question A resolves NO")
    assert symbolic_statement("(NOT P) OR (Q)").startswith("(NOT A) OR (B)")


def test_strip_creator_edits():
    text = ("Resolves YES if X.\n\nUpdate 2025-10-01 (PST) (AI summary of creator comment): it happened.\n\n"
            "[image]Update 2025-10-02 (PST): more.\n\nMore rules. Edit Jun 16: outcome hint\n\nEDIT: final")
    out, removed = strip_creator_edits(text)
    assert out == "Resolves YES if X.\n\nMore rules."
    assert removed == 4


def test_v7_explicit_statements():
    from training.verbalized_baseline import explicit_statement, question_block_v7
    binary = {"parent_question": "Will X win?", "answer_text": None, "closing_week": "2025-12-01",
              "criteria_text": "[image]"}
    option = {"parent_question": "Who wins?", "answer_text": "Chelsea", "closing_week": "2025-12-01",
              "criteria_text": ""}
    assert explicit_statement("NOT P", binary, option, negation_note=True) == (
        '(a NEGATION) The answer to Question A ("Will X win?") is NOT yes: it is NO. '
        "This statement is TRUE if the answer is NO, and FALSE if the answer is YES.")
    assert explicit_statement("P", binary, option, negation_note=True) == 'The answer to Question A ("Will X win?") is YES.'
    assert "will NOT happen" in explicit_statement("NOT Q", binary, option)
    assert explicit_statement("(P) AND (NOT Q)", binary, option).startswith("BOTH of these are true: (1) the answer")
    block = question_block_v7(binary, "A")
    assert "resolves YES" not in block and "[image]" not in block
    assert block.endswith("Resolution criteria: not available (see the question).")
