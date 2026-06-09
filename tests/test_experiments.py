from product_intelligence.experimentation.assignment import assign_variant


def test_assignment_is_deterministic():
    a = assign_variant("acct_1", "exp_a")
    b = assign_variant("acct_1", "exp_a")
    assert a.variant == b.variant


def test_assignment_respects_weights():
    counts = {"control": 0, "treatment": 0}
    for i in range(4000):
        v = assign_variant(f"u{i}", "exp_a", {"control": 0.8, "treatment": 0.2}).variant
        counts[v] += 1
    treat_share = counts["treatment"] / 4000
    assert 0.16 < treat_share < 0.24


def test_independent_across_experiments():
    # Different experiments should not be perfectly correlated for the same unit.
    diffs = sum(
        assign_variant(f"u{i}", "exp_a").variant != assign_variant(f"u{i}", "exp_b").variant
        for i in range(200)
    )
    assert diffs > 0
