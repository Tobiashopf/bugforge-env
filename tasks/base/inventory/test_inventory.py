import pytest
from inventory import Inventory


def make():
    inv = Inventory()
    inv.add("B2", 4, 2.5, "Tools")
    inv.add("A1", 10, 1.0, "tools")
    inv.add("C3", 2, 20.0, "Garden")
    return inv


def test_add_sets_quantity():
    inv = Inventory()
    inv.add("A1", 3, 10.0, "tools")
    assert inv.quantity("A1") == 3


def test_add_existing_sku_accumulates():
    inv = Inventory()
    inv.add("A1", 3, 10.0, "tools")
    inv.add("A1", 2, 10.0, "tools")
    assert inv.quantity("A1") == 5


def test_add_quantity_one_is_allowed():
    inv = Inventory()
    inv.add("A1", 1, 10.0, "tools")
    assert inv.quantity("A1") == 1


def test_add_zero_quantity_raises():
    with pytest.raises(ValueError):
        Inventory().add("A1", 0, 10.0, "tools")


def test_add_negative_quantity_raises():
    with pytest.raises(ValueError):
        Inventory().add("A1", -2, 10.0, "tools")


def test_add_zero_price_is_allowed():
    inv = Inventory()
    inv.add("A1", 2, 0.0, "tools")
    assert inv.total_value() == 0.0


def test_add_negative_price_raises():
    with pytest.raises(ValueError):
        Inventory().add("A1", 1, -0.01, "tools")


def test_add_price_mismatch_raises():
    inv = Inventory()
    inv.add("A1", 1, 10.0, "tools")
    with pytest.raises(ValueError):
        inv.add("A1", 1, 11.0, "tools")


def test_failed_add_leaves_stock_unchanged():
    inv = Inventory()
    inv.add("A1", 3, 10.0, "tools")
    with pytest.raises(ValueError):
        inv.add("A1", 1, 9.0, "tools")
    assert inv.quantity("A1") == 3


def test_quantity_unknown_sku_is_zero():
    assert Inventory().quantity("nope") == 0


def test_remove_partial_decreases_quantity():
    inv = make()
    inv.remove("A1", 3)
    assert inv.quantity("A1") == 7


def test_remove_all_deletes_sku():
    inv = make()
    inv.remove("C3", 2)
    assert inv.quantity("C3") == 0
    assert "C3" not in inv.low_stock(limit=100)


def test_remove_one_less_than_stock_keeps_one():
    inv = make()
    inv.remove("B2", 3)
    assert inv.quantity("B2") == 1


def test_remove_more_than_stock_raises():
    inv = make()
    with pytest.raises(ValueError):
        inv.remove("C3", 3)


def test_remove_zero_raises():
    inv = make()
    with pytest.raises(ValueError):
        inv.remove("A1", 0)


def test_remove_unknown_sku_raises_key_error():
    with pytest.raises(KeyError):
        Inventory().remove("nope", 1)


def test_total_value_below_threshold_no_discount():
    inv = Inventory()
    inv.add("A1", 3, 33.33, "x")
    assert inv.total_value() == 99.99


def test_total_value_exactly_at_threshold_gets_discount():
    inv = Inventory()
    inv.add("A1", 4, 25.0, "x")
    assert inv.total_value() == 90.0


def test_total_value_above_threshold_gets_discount():
    inv = Inventory()
    inv.add("A1", 2, 60.0, "x")
    assert inv.total_value() == 108.0


def test_total_value_sums_all_items():
    assert make().total_value() == 60.0


def test_total_value_custom_threshold_and_rate():
    assert make().total_value(discount_threshold=60.0, discount_rate=0.25) == 45.0


def test_total_value_rounds_to_two_decimals():
    inv = Inventory()
    inv.add("A1", 2, 0.3333, "x")
    assert inv.total_value() == 0.67


def test_total_value_empty_is_zero():
    assert Inventory().total_value() == 0.0


def test_by_category_is_case_insensitive_and_sorted():
    assert make().by_category("TOOLS") == ["A1", "B2"]


def test_by_category_unknown_is_empty():
    assert make().by_category("kitchen") == []


def test_low_stock_default_limit_is_inclusive():
    inv = Inventory()
    inv.add("A", 5, 1.0, "x")
    inv.add("B", 6, 1.0, "x")
    assert inv.low_stock() == ["A"]


def test_low_stock_orders_by_quantity():
    assert make().low_stock(limit=10) == ["C3", "B2", "A1"]


def test_low_stock_ties_broken_by_sku():
    inv = Inventory()
    inv.add("Z", 1, 1.0, "x")
    inv.add("M", 1, 1.0, "x")
    inv.add("K", 3, 1.0, "x")
    assert inv.low_stock(limit=2) == ["M", "Z"]
