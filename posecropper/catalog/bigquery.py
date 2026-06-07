"""Production catalog lookup against two BigQuery catalog views with an EAN fallback.
Kept to show the real engineering; the offline demo uses StaticCategoryProvider instead.

NOTE: queries are interpolated as in the original system. Production hardening would use
parameterized queries; the genericized view names are config-driven."""
import os

from google.cloud import bigquery

from posecropper.catalog.base import CategoryProvider


class BigQueryCategoryProvider(CategoryProvider):
    def __init__(self, project_id: str | None = None,
                 view_a: str | None = None, view_b: str | None = None):
        self.project_id = project_id or os.environ.get("PROJECT_ID", "your-gcp-project-id")
        self.view_a = view_a or os.environ.get("CATALOG_VIEW_A", "catalog_views.item_catalog_a")
        self.view_b = view_b or os.environ.get("CATALOG_VIEW_B", "catalog_views.item_catalog_b")

    def _query_first(self, client, field, value):
        for view in (self.view_a, self.view_b):  # catalog A first, then catalog B
            sql = f"SELECT StyleCode, ItemGroupName FROM `{view}` WHERE {field} = '{value}'"
            df = client.query(sql).result().to_dataframe()
            if not df.empty:
                return df
        return None

    def get_item_group(self, style_code: str | None) -> str | None:
        if not style_code:
            return None
        try:
            client = bigquery.Client(project=self.project_id)
            df = self._query_first(client, "StyleCode", style_code)
            if df is None:
                df = self._query_first(client, "EAN", style_code)  # fallback by EAN
            if df is None or df.empty:
                return None
            return df["ItemGroupName"].iloc[0]
        except Exception as e:  # noqa: BLE001
            print(f"Error querying catalog for {style_code}: {e}")
            return None
