"""Offline mock: return a fixed ItemGroupName (or a per-code mapping). Lets the demo
process an image end to end by passing the garment category directly — no GCP."""
from posecropper.catalog.base import CategoryProvider


class StaticCategoryProvider(CategoryProvider):
    def __init__(self, item_group: str | None = None, mapping: dict | None = None):
        self.item_group = item_group
        self.mapping = mapping or {}

    def get_item_group(self, style_code: str | None) -> str | None:
        if style_code in self.mapping:
            return self.mapping[style_code]
        return self.item_group
