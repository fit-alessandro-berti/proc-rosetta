from proc_rosetta.tokenizers import TreeTokenizer
from proc_rosetta.tree import ProcessTreeNode


def test_tree_canonicalizes_commutative_children_and_activity_names():
    tree = ProcessTreeNode.xor(ProcessTreeNode.activity("z"), ProcessTreeNode.activity("a"))

    canonical = tree.canonicalize_activity_labels()

    assert canonical.activity_labels() == ("A0", "A1")
    assert str(ProcessTreeNode.xor(ProcessTreeNode.activity("b"), ProcessTreeNode.activity("a"))).startswith(
        "XOR("
    )


def test_tree_tokenizer_round_trip_and_grammar_mask():
    tokenizer = TreeTokenizer(max_activities=4, max_arity=3)
    tree = ProcessTreeNode.seq(
        ProcessTreeNode.activity("raw_a"),
        ProcessTreeNode.xor(ProcessTreeNode.activity("raw_b"), ProcessTreeNode.activity("raw_a")),
    )

    encoded = tokenizer.encode_tree(tree)
    decoded = tokenizer.decode_tree(encoded)

    assert decoded.to_dict() == tree.canonicalize_activity_labels().to_dict()
    mask_after_bos = tokenizer.next_token_mask([tokenizer.bos_id])
    assert mask_after_bos[tokenizer.token_to_id["SEQ"]]
    assert not mask_after_bos[tokenizer.token_to_id["ARITY_2"]]


def test_tree_reassociates_wide_operators_for_a_bounded_tokenizer():
    tree = ProcessTreeNode.seq(
        *(ProcessTreeNode.activity(label) for label in ("a", "b", "c", "d", "e"))
    )
    normalized = tree.reassociate_operators(3)

    def maximum_arity(node):
        return max([len(node.children), *(maximum_arity(child) for child in node.children)])

    assert normalized.activity_labels() == tree.activity_labels()
    assert maximum_arity(normalized) == 3
    TreeTokenizer(max_activities=5, max_arity=3).encode_tree(normalized)


def test_tree_tokenizer_masks_loop_as_two_child_operator():
    tokenizer = TreeTokenizer(max_activities=4, max_arity=4)
    prefix = [tokenizer.bos_id, tokenizer.token_to_id["LOOP"]]

    mask = tokenizer.next_token_mask(prefix)

    assert mask[tokenizer.token_to_id["ARITY_2"]]
    assert not mask[tokenizer.token_to_id["ARITY_3"]]

    tree = ProcessTreeNode.loop(
        ProcessTreeNode.activity("raw_a"),
        ProcessTreeNode.activity("raw_b"),
    )
    decoded = tokenizer.decode_tree(tokenizer.encode_tree(tree))
    assert len(decoded.children) == 2
