"""A small stock inventory with pricing and discount rules."""


class Inventory:
    """Tracks quantity, unit price and category for each SKU."""

    def __init__(self):
        self._items = {}

    def add(self, sku, qty, unit_price, category):
        """Add stock. Re-adding an existing SKU must use the same price."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        if unit_price < 0:
            raise ValueError("unit_price must be non-negative")
        if sku in self._items:
            item = self._items[sku]
            if item["price"] != unit_price:
                raise ValueError("price mismatch for existing sku")
            item["qty"] += qty
        else:
            self._items[sku] = {"qty": qty, "price": unit_price, "category": category}

    def remove(self, sku, qty):
        """Remove stock. Removing the whole quantity deletes the SKU."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        if sku not in self._items:
            raise KeyError(sku)
        item = self._items[sku]
        if qty > item["qty"]:
            raise ValueError("not enough stock")
        if qty == item["qty"]:
            del self._items[sku]
        else:
            item["qty"] -= qty

    def quantity(self, sku):
        """Units in stock, or 0 for an unknown SKU."""
        if sku in self._items:
            return self._items[sku]["qty"]
        return 0

    def total_value(self, discount_threshold=100.0, discount_rate=0.10):
        """Stock value; the discount applies when the subtotal reaches the threshold."""
        subtotal = 0.0
        for item in self._items.values():
            subtotal += item["qty"] * item["price"]
        if subtotal >= discount_threshold:
            subtotal = subtotal * (1 - discount_rate)
        return round(subtotal, 2)

    def by_category(self, category):
        """SKUs in a category (case-insensitive), sorted alphabetically."""
        wanted = category.lower()
        found = []
        for sku, item in self._items.items():
            if item["category"].lower() == wanted:
                found.append(sku)
        return sorted(found)

    def low_stock(self, limit=5):
        """SKUs with quantity at or below the limit, by quantity then SKU."""
        rows = []
        for sku, item in self._items.items():
            if item["qty"] <= limit:
                rows.append((item["qty"], sku))
        rows.sort()
        return [sku for _, sku in rows]
