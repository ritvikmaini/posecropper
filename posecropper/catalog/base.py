"""Catalog lookup behind an interface so the demo can run offline (StaticCategoryProvider)
while the production BigQuery implementation remains visible (BigQueryCategoryProvider)."""
from abc import ABC, abstractmethod


class CategoryProvider(ABC):
    @abstractmethod
    def get_item_group(self, style_code: str | None) -> str | None:
        """Resolve a style/model code to its catalog ItemGroupName, or None if unknown."""
