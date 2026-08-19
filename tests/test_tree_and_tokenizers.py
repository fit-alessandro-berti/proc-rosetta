from proc_rosetta.tokenizers import ActivityTokenizer, TreeTokenizer
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


def test_tree_normalization_flattens_associative_syntax_idempotently():
    left = ProcessTreeNode.seq(
        ProcessTreeNode.activity("a"),
        ProcessTreeNode.seq(
            ProcessTreeNode.activity("b"), ProcessTreeNode.activity("c")
        ),
    )
    right = ProcessTreeNode.seq(
        ProcessTreeNode.seq(
            ProcessTreeNode.activity("a"), ProcessTreeNode.activity("b")
        ),
        ProcessTreeNode.activity("c"),
    )

    assert left.normalize(3).to_dict() == right.normalize(3).to_dict()
    assert left.normalize(3).normalize(3).to_dict() == left.normalize(3).to_dict()


def test_unknown_activity_has_a_trainable_non_padding_id():
    tokenizer = ActivityTokenizer(max_activities=3)

    assert tokenizer.unk_id != tokenizer.pad_id
    assert tokenizer.encode_trace(["outside"])[0] == tokenizer.unk_id


def test_tree_tokenizer_masks_and_round_trips_three_child_loop():
    tokenizer = TreeTokenizer(max_activities=4, max_arity=4)
    prefix = [tokenizer.bos_id, tokenizer.token_to_id["LOOP"]]

    mask = tokenizer.next_token_mask(prefix)

    assert mask[tokenizer.token_to_id["ARITY_2"]]
    assert mask[tokenizer.token_to_id["ARITY_3"]]

    tree = ProcessTreeNode.loop(
        ProcessTreeNode.activity("raw_a"),
        ProcessTreeNode.activity("raw_b"),
        ProcessTreeNode.activity("raw_c"),
    )
    decoded = tokenizer.decode_tree(tokenizer.encode_tree(tree))
    assert len(decoded.children) == 3
