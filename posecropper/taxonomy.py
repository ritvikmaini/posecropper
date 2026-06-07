"""Crop-geometry constants and the garment-category taxonomy.

The article lists map a catalog's ItemGroupName to one of three routing buckets.
They are a generic, bilingual (DE/EN) apparel + accessory set — representative of
a German fashion e-commerce catalog, with no employer-identifying entries.
"""

# --- Crop geometry constants (target catalog aspect ratio height:width = 1.33) ---
ASPECT_RATIO = 1.33
VERTICAL_TOP_PADDING = 0.12
VERTICAL_BOTTOM_PADDING = 0.10
EXTRA_ABOVE_CENTER_RATIO = 0.40
EXTRA_BELOW_CENTER_RATIO = 0.20

# Classical-CV product path (disabled/explored — see product_detection.py)
MIN_AREA_RATIO = 0.12
MAX_AREA_RATIO = 0.97
PRODUCT_PADDING = 0.10

# --- Garment-category taxonomy (bilingual DE/EN) ---
UPPER_HALF_ARTICLES = [
    # EN
    "T-Shirts", "Shirts", "Longsleeves", "Polos", "Tank Tops", "Tops",
    "Hoodies", "Sweatshirts", "Cardigans", "Knitwear", "Blouses",
    "Jackets", "Coats", "Blazers", "Vests", "Dresses",
    # DE
    "Hemden", "Langarmshirts", "Poloshirts", "Tanktops", "Kapuzenpullover",
    "Pullover", "Strickjacken", "Strickwaren", "Blusen", "Jacken",
    "Maentel", "Sakkos", "Westen", "Kleider",
]

LOWER_HALF_ARTICLES = [
    # EN
    "Jeans", "Trousers", "Pants", "Chinos", "Shorts", "Skirts",
    "Leggings", "Joggers", "Cargo Pants",
    # DE
    "Hosen", "Stoffhosen", "Roecke", "Jogginghosen", "Cargohosen",
]

REVIEW_ARTICLES = [
    # EN
    "Caps", "Hats", "Beanies", "Scarves", "Gloves", "Belts", "Socks",
    "Bags", "Backpacks", "Sunglasses", "Jewellery", "Shoes", "Sneakers",
    "Boots", "Swimwear", "Underwear",
    # DE
    "Muetzen", "Huete", "Schals", "Handschuhe", "Guertel", "Socken",
    "Taschen", "Rucksaecke", "Sonnenbrillen", "Schmuck", "Schuhe",
    "Stiefel", "Bademode", "Unterwaesche",
]
