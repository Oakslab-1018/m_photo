"""Pure-stdlib contract for experimental city scene-condition matching.

This module is shared by the StreetCLIP inference service and the HTTP
gateway. It defines a bounded vocabulary in which every caller value maps to a
fixed canonical token, English image/text description and Chinese display
label. Callers can never inject prompt text.

The resulting similarities are experimental scene-condition model matches
between an image and fixed descriptions combined with catalog city/country
names. They are not city-attribute verification, do not establish that a city
has a particular building, terrain, water, vegetation, road, vehicle, climate
or infrastructure feature in general, and carry no validated accuracy claim.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

CITY_SCENE_VERSION = "streetclip-city-scene-prompts-v1"
MAX_CITY_SCENE_CITIES = 20
MAX_CITY_SCENE_CLUES = 6
MAX_CITY_SCENE_PROMPTS = MAX_CITY_SCENE_CITIES * MAX_CITY_SCENE_CLUES * 2

# The dimension vocabulary deliberately excludes language and driving_side:
# those country-level signals are not meaningful same-country city differences.
CITY_SCENE_DIMENSIONS: tuple[str, ...] = (
    "building",
    "terrain",
    "waterbody",
    "vegetation",
    "road",
    "vehicle",
    "climate",
    "infrastructure",
)

CITY_SCENE_PROMPT_TEMPLATES: tuple[str, str] = (
    "a street-level photograph of {city}, {country}, showing {description}",
    "a street scene in {city}, {country}, with {description}",
)


@dataclass(frozen=True, slots=True)
class CitySceneValue:
    """One fixed city-scene condition with no caller-controlled prompt text."""

    dimension: str
    value: str
    label: str
    description: str


# Canonical values follow inference/streetclip/scene.py for the six shared
# dimensions, except that the scene dimension ``water`` is named ``waterbody``.
# Climate and infrastructure use the bounded tokens already present in
# services/chat.py. Each English row is a fixed visual condition description.
_VALUE_ROWS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "building": (
        ("high_rise", "高层建筑", "tall high-rise towers"),
        ("mid_rise", "中层建筑", "mid-rise apartment or office buildings"),
        ("low_rise", "低层建筑", "low-rise houses or small buildings"),
        ("historic_masonry", "历史砖石建筑", "historic brick or stone architecture"),
        ("commercial_storefront", "商业店面", "commercial storefronts along the street"),
        ("industrial_warehouse", "工业或仓储建筑", "industrial buildings or warehouses"),
        ("brick_masonry", "红砖或砖石建筑", "visible red brick or masonry walls"),
        ("traditional_architecture", "传统建筑场景", "traditional local architecture visible in the street scene"),
        ("modern_glass", "现代玻璃幕墙建筑", "modern buildings with glass curtain walls"),
        ("generic", "普通城市建筑", "ordinary urban buildings along the street"),
        ("gothic", "哥特式建筑", "gothic-style architecture"),
        ("colonial", "殖民风格建筑", "colonial-style architecture"),
        ("mediterranean", "地中海风格建筑", "Mediterranean-style architecture"),
    ),
    "terrain": (
        ("flat_ground", "平坦地面", "visibly flat level ground"),
        ("sloped_ground", "坡地", "a noticeable street slope or incline"),
        ("hilly", "丘陵", "rolling hills around the street"),
        ("mountainous", "山地", "steep mountainous terrain"),
        ("valley", "谷地", "a valley between elevated terrain"),
        ("plateau", "高原", "elevated plateau terrain"),
        ("coastal_lowland", "沿海低地", "low coastal terrain"),
    ),
    "waterbody": (
        ("no_visible_water", "未见水体", "no visible body of water"),
        ("river", "河流", "a river beside the street"),
        ("lake", "湖泊", "a lake beside the street"),
        ("sea_ocean", "海洋", "the sea or ocean beside the street"),
        ("canal", "运河", "a canal beside the street"),
        ("harbor", "港口", "a harbor or dock area beside the street"),
        ("waterfall", "瀑布", "a waterfall visible from the street"),
        ("pond_fountain", "水池或喷泉", "a small pond or fountain in the street scene"),
    ),
    "vegetation": (
        ("street_trees", "行道树", "street trees lining the road"),
        ("dense_green", "茂密绿地", "dense green vegetation around the street"),
        ("sparse_green", "稀疏植被", "only sparse vegetation around the street"),
        ("grass_lawn", "草地或草坪", "a grassy lawn or field beside the street"),
        ("flowering_plants", "花卉植物", "flowering plants or shrubs beside the street"),
        ("dry_vegetation", "干燥植被", "dry scrub or arid vegetation beside the street"),
        ("tropical_rainforest", "热带雨林", "lush tropical rainforest vegetation beside the street"),
        ("temperate_forest", "温带森林", "temperate broadleaf or conifer forest beside the street"),
        ("grassland", "草原", "open grassland beside the street"),
        ("savanna", "稀树草原", "a savanna with scattered trees beside the street"),
        ("desert", "荒漠", "desert or arid scrub vegetation beside the street"),
        ("urban_sparse", "城市稀疏植被", "sparse urban vegetation around the street"),
    ),
    "road": (
        ("arterial_road", "主干道", "a wide multi-lane urban arterial road"),
        ("local_street", "城市街道", "a narrow city street between buildings"),
        ("residential_lane", "住宅道路", "a quiet residential lane between homes"),
        ("highway", "高速公路", "a highway or motorway photographed from the roadside"),
        ("unpaved_road", "土路或碎石路", "an unpaved dirt or gravel road"),
        ("pedestrian_way", "步行道", "a paved pedestrian sidewalk or walkway"),
        ("asphalt", "沥青路面", "an asphalt-paved street surface"),
        ("concrete", "水泥路面", "a concrete-paved street surface"),
        ("cobblestone", "石板路面", "a cobblestone-paved street"),
        ("paved", "铺装路面", "a clearly paved street surface"),
        ("unpaved", "未铺装路面", "an unpaved road surface"),
        ("dirt", "土路路面", "a dirt road surface"),
        ("gravel", "碎石路面", "a gravel road surface"),
    ),
    "vehicle": (
        ("cars", "汽车", "passenger cars on the street"),
        ("motorcycles_scooters", "摩托车或踏板车", "motorcycles or scooters on the street"),
        ("bicycles", "自行车", "bicycles on or beside the street"),
        ("buses", "公交车", "a public bus on the street"),
        ("trucks", "卡车", "a truck or lorry on the street"),
        ("no_visible_vehicles", "未见车辆", "no visible vehicles on the street"),
        ("dark_vehicle", "深色车辆", "dark-colored passenger vehicles on the street"),
        ("white_silver_vehicle", "白色或银色车辆", "white or silver passenger vehicles on the street"),
        ("unknown_vehicle", "未识别车辆", "vehicles of uncertain type on the street"),
    ),
    "climate": (
        ("tropical", "热带气候", "warm tropical conditions"),
        ("tropical_humid", "热带湿润气候", "warm and humid tropical conditions"),
        ("humid", "湿润气候", "a visibly humid and moisture-rich environment"),
        ("temperate", "温带气候", "mild temperate conditions"),
        ("mediterranean_sunny", "地中海晴朗气候", "bright Mediterranean-type sunny conditions"),
        ("temperate_maritime", "温带海洋性气候", "cool and damp maritime conditions"),
        ("cold_polar", "寒冷或极地气候", "cold polar or subpolar conditions"),
        ("arid", "干旱气候", "dry arid conditions"),
    ),
    "infrastructure": (
        ("railway", "铁路", "railway tracks or a train"),
        ("subway", "地铁", "a subway or metro entrance or train"),
        ("tram", "有轨电车", "tram tracks or a tram"),
        ("power_lines", "输电线路", "overhead power lines or utility poles"),
        ("wind_turbine", "风力发电机", "wind turbines"),
        ("bridge", "桥梁", "a bridge"),
        ("tunnel", "隧道", "a road tunnel"),
        ("airport", "机场", "airport infrastructure or aircraft"),
    ),
}

CITY_SCENE_VALUES: dict[str, tuple[CitySceneValue, ...]] = {
    dimension: tuple(
        CitySceneValue(
            dimension=dimension,
            value=value,
            label=label,
            description=description,
        )
        for value, label, description in rows
    )
    for dimension, rows in _VALUE_ROWS.items()
}
CITY_SCENE_VALUE_BY_DIMENSION_VALUE: dict[tuple[str, str], CitySceneValue] = {
    (item.dimension, item.value): item
    for values in CITY_SCENE_VALUES.values()
    for item in values
}
CITY_SCENE_VALUE_LABELS: dict[tuple[str, str], str] = {
    key: item.label
    for key, item in CITY_SCENE_VALUE_BY_DIMENSION_VALUE.items()
}
CITY_SCENE_VALUE_DESCRIPTIONS: dict[tuple[str, str], str] = {
    key: item.description
    for key, item in CITY_SCENE_VALUE_BY_DIMENSION_VALUE.items()
}

# Exact, fixed aliases only. No substring, fuzzy or language/driving-side
# inference is performed. Unknown values return ``None`` from the normalizer.
_ALIASES: dict[tuple[str, str], str] = {
    # English tokens emitted or understood by services/chat.py today.
    ("terrain", "mountain"): "mountainous",
    ("terrain", "hills"): "hilly",
    ("terrain", "plain"): "flat_ground",
    ("terrain", "coast"): "coastal_lowland",
    ("waterbody", "sea"): "sea_ocean",
    ("waterbody", "ocean"): "sea_ocean",
    # Chinese building observations.
    ("building", "highrise"): "high_rise",
    ("building", "brick"): "brick_masonry",
    ("building", "red_brick"): "brick_masonry",
    ("building", "brick_wall"): "brick_masonry",
    ("building", "traditional"): "traditional_architecture",
    ("building", "glass_curtain_wall"): "modern_glass",
    ("building", "高楼"): "high_rise",
    ("building", "高层"): "high_rise",
    ("building", "摩天楼"): "high_rise",
    ("building", "现代建筑"): "modern_glass",
    ("building", "玻璃幕墙"): "modern_glass",
    ("building", "玻璃建筑"): "modern_glass",
    ("building", "玻璃大楼"): "modern_glass",
    ("building", "中层楼"): "mid_rise",
    ("building", "矮楼"): "low_rise",
    ("building", "低层"): "low_rise",
    ("building", "传统建筑"): "traditional_architecture",
    ("building", "传统"): "traditional_architecture",
    ("building", "中式"): "traditional_architecture",
    ("building", "中式建筑"): "traditional_architecture",
    ("building", "传统民居"): "traditional_architecture",
    ("building", "传统街区"): "traditional_architecture",
    ("building", "古建筑"): "traditional_architecture",
    ("building", "老建筑"): "traditional_architecture",
    ("building", "历史建筑"): "traditional_architecture",
    ("building", "红砖"): "brick_masonry",
    ("building", "红砖建筑"): "brick_masonry",
    ("building", "红砖墙"): "brick_masonry",
    ("building", "砖砌建筑"): "brick_masonry",
    ("building", "砖墙"): "brick_masonry",
    ("building", "砖石墙"): "brick_masonry",
    ("building", "砖石建筑"): "brick_masonry",
    ("building", "哥特式"): "gothic",
    ("building", "哥特"): "gothic",
    ("building", "殖民风格"): "colonial",
    ("building", "殖民地风格"): "colonial",
    ("building", "地中海风格"): "mediterranean",
    ("building", "地中海建筑"): "mediterranean",
    ("building", "商店街"): "commercial_storefront",
    ("building", "仓库"): "industrial_warehouse",
    # Chinese road observations.
    ("road", "主干路"): "arterial_road",
    ("road", "小街"): "local_street",
    ("road", "小巷"): "residential_lane",
    ("road", "住宅区道路"): "residential_lane",
    ("road", "高速路"): "highway",
    ("road", "高速公路"): "highway",
    ("road", "未铺装"): "unpaved",
    ("road", "土路"): "dirt",
    ("road", "泥路"): "dirt",
    ("road", "砂石路"): "gravel",
    ("road", "碎石路"): "gravel",
    ("road", "沥青"): "asphalt",
    ("road", "柏油"): "asphalt",
    ("road", "水泥路"): "concrete",
    ("road", "水泥路面"): "concrete",
    ("road", "石板路"): "cobblestone",
    ("road", "石板"): "cobblestone",
    ("road", "铺装"): "paved",
    ("road", "人行道"): "pedestrian_way",
    ("road", "步行街"): "pedestrian_way",
    # Chinese vegetation observations.
    ("vegetation", "树林"): "dense_green",
    ("vegetation", "森林"): "temperate_forest",
    ("vegetation", "温带森林"): "temperate_forest",
    ("vegetation", "阔叶林"): "temperate_forest",
    ("vegetation", "针叶林"): "temperate_forest",
    ("vegetation", "雨林"): "tropical_rainforest",
    ("vegetation", "热带雨林"): "tropical_rainforest",
    ("vegetation", "茂密植被"): "dense_green",
    ("vegetation", "植被稀少"): "urban_sparse",
    ("vegetation", "城市稀疏植被"): "urban_sparse",
    ("vegetation", "草原"): "grassland",
    ("vegetation", "草地"): "grassland",
    ("vegetation", "牧场"): "grassland",
    ("vegetation", "稀树草原"): "savanna",
    ("vegetation", "荒漠"): "desert",
    ("vegetation", "沙漠"): "desert",
    ("vegetation", "戈壁"): "desert",
    # Chinese terrain observations.
    ("terrain", "山地"): "mountainous",
    ("terrain", "山脉"): "mountainous",
    ("terrain", "群山"): "mountainous",
    ("terrain", "山区"): "mountainous",
    ("terrain", "雪山"): "mountainous",
    ("terrain", "丘陵"): "hilly",
    ("terrain", "高原"): "plateau",
    ("terrain", "坡地"): "sloped_ground",
    ("terrain", "斜坡"): "sloped_ground",
    ("terrain", "平原"): "flat_ground",
    ("terrain", "盆地"): "valley",
    ("terrain", "山谷"): "valley",
    ("terrain", "海岸"): "coastal_lowland",
    ("terrain", "海滨"): "coastal_lowland",
    ("terrain", "沿海"): "coastal_lowland",
    # Chinese water observations.
    ("waterbody", "河流"): "river",
    ("waterbody", "河边"): "river",
    ("waterbody", "江"): "river",
    ("waterbody", "水库"): "lake",
    ("waterbody", "湖泊"): "lake",
    ("waterbody", "湖边"): "lake",
    ("waterbody", "海洋"): "sea_ocean",
    ("waterbody", "大海"): "sea_ocean",
    ("waterbody", "海边"): "sea_ocean",
    ("waterbody", "海滩"): "sea_ocean",
    ("waterbody", "港口"): "harbor",
    ("waterbody", "码头"): "harbor",
    ("waterbody", "瀑布"): "waterfall",
    ("waterbody", "运河"): "canal",
    ("waterbody", "水池"): "pond_fountain",
    ("waterbody", "喷泉"): "pond_fountain",
    # Chinese vehicle observations.
    ("vehicle", "小汽车"): "cars",
    ("vehicle", "轿车"): "cars",
    ("vehicle", "摩托"): "motorcycles_scooters",
    ("vehicle", "摩托车"): "motorcycles_scooters",
    ("vehicle", "踏板车"): "motorcycles_scooters",
    ("vehicle", "电动车"): "motorcycles_scooters",
    ("vehicle", "单车"): "bicycles",
    ("vehicle", "bicycle"): "bicycles",
    ("vehicle", "自行车"): "bicycles",
    ("vehicle", "公交"): "buses",
    ("vehicle", "巴士"): "buses",
    ("vehicle", "bus"): "buses",
    ("vehicle", "公共汽车"): "buses",
    ("vehicle", "公交车"): "buses",
    ("vehicle", "货车"): "trucks",
    ("vehicle", "motorcycle"): "motorcycles_scooters",
    ("vehicle", "scooter"): "motorcycles_scooters",
    ("vehicle", "卡车"): "trucks",
    # Chinese climate observations.
    ("climate", "热带"): "tropical",
    ("climate", "热带气候"): "tropical",
    ("climate", "热带湿润"): "tropical_humid",
    ("climate", "湿润"): "humid",
    ("climate", "潮湿"): "humid",
    ("climate", "多雨"): "humid",
    ("climate", "湿润气候"): "humid",
    ("climate", "温和"): "temperate",
    ("climate", "温带"): "temperate",
    ("climate", "地中海"): "mediterranean_sunny",
    ("climate", "地中海气候"): "mediterranean_sunny",
    ("climate", "温带海洋"): "temperate_maritime",
    ("climate", "海洋性气候"): "temperate_maritime",
    ("climate", "温带海洋性气候"): "temperate_maritime",
    ("climate", "寒冷"): "cold_polar",
    ("climate", "寒冷气候"): "cold_polar",
    ("climate", "极地"): "cold_polar",
    ("climate", "寒带"): "cold_polar",
    ("climate", "干旱"): "arid",
    ("climate", "干旱气候"): "arid",
    ("climate", "干燥"): "arid",
    # Chinese infrastructure observations.
    ("infrastructure", "铁路"): "railway",
    ("infrastructure", "火车"): "railway",
    ("infrastructure", "铁道"): "railway",
    ("infrastructure", "地铁"): "subway",
    ("infrastructure", "有轨电车"): "tram",
    ("infrastructure", "电车"): "tram",
    ("infrastructure", "输电线路"): "power_lines",
    ("infrastructure", "高压线"): "power_lines",
    ("infrastructure", "电线杆"): "power_lines",
    ("infrastructure", "风力发电机"): "wind_turbine",
    ("infrastructure", "风力发电"): "wind_turbine",
    ("infrastructure", "桥梁"): "bridge",
    ("infrastructure", "大桥"): "bridge",
    ("infrastructure", "隧道"): "tunnel",
    ("infrastructure", "机场"): "airport",
}

# Exact dimension aliases retained for compatibility with the older fixed scene
# vocabulary. New callers should send ``waterbody``.
_DIMENSION_ALIASES: dict[str, str] = {"water": "waterbody"}


def _normalize_token(value: str) -> str:
    """Normalize only case, whitespace and simple separators."""
    text = " ".join(value.strip().casefold().split())
    return text.replace("-", "_").replace(" ", "_")


def normalize_city_scene_clue(
    dimension: str,
    value: str,
) -> tuple[str, str] | None:
    """Return a canonical ``(dimension, value)`` or ``None`` if unknown.

    Matching is exact after separator/case normalization. Display labels and
    the explicit fixed aliases above are accepted. The function never guesses
    from substrings, never accepts prompt text and never treats
    ``language``/``driving_side`` as same-country city differences.
    """

    if not isinstance(dimension, str) or not isinstance(value, str):
        return None
    dimension_token = _normalize_token(dimension)
    dimension_token = _DIMENSION_ALIASES.get(dimension_token, dimension_token)
    value_token = _normalize_token(value)
    if not dimension_token or not value_token:
        return None

    if (dimension_token, value_token) in CITY_SCENE_VALUE_BY_DIMENSION_VALUE:
        return dimension_token, value_token

    alias_target = _ALIASES.get((dimension_token, value_token))
    if alias_target is not None:
        return dimension_token, alias_target

    # The fixed Chinese display labels are themselves stable aliases.
    for item in CITY_SCENE_VALUES.get(dimension_token, ()):
        if _normalize_token(item.label) == value_token:
            return item.dimension, item.value
    return None


def validate_city_scene_clues(
    raw: Any,
) -> tuple[tuple[str, str], ...]:
    """Validate a decoded JSON clue list and return canonical pairs.

    Input must be a list/tuple of 1 to 6 objects containing exactly
    ``dimension`` and ``value`` string fields. Unknown dimensions or values are
    rejected rather than passed through as free text. Multiple values for one
    dimension are intentionally allowed in input order; conflict resolution is
    a backend policy decision, not part of this shared vocabulary contract.
    """

    if not isinstance(raw, (list, tuple)):
        raise ValueError("clues must be a JSON array")
    if not 1 <= len(raw) <= MAX_CITY_SCENE_CLUES:
        raise ValueError(
            f"clues must contain 1 to {MAX_CITY_SCENE_CLUES} entries"
        )
    normalized: list[tuple[str, str]] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ValueError("each clue must be an object")
        if set(entry) != {"dimension", "value"}:
            raise ValueError("each clue must contain only dimension and value")
        dimension = entry.get("dimension")
        value = entry.get("value")
        if not isinstance(dimension, str) or not isinstance(value, str):
            raise ValueError("clue dimension and value must be strings")
        pair = normalize_city_scene_clue(dimension, value)
        if pair is None:
            raise ValueError("unknown clue dimension or value")
        normalized.append(pair)
    return tuple(normalized)


def city_scene_value(dimension: str, value: str) -> CitySceneValue | None:
    """Return the fixed label/description entry for one canonical pair."""
    pair = normalize_city_scene_clue(dimension, value)
    if pair is None:
        return None
    return CITY_SCENE_VALUE_BY_DIMENSION_VALUE.get(pair)


def render_city_scene_prompts(
    city_name: str,
    country_name: str,
    clue: tuple[str, str] | CitySceneValue,
) -> tuple[str, str]:
    """Render exactly two fixed prompts containing the original city/country.

    The arguments must already come from the catalog and validated clue
    contract. This helper performs no catalog lookup, model call or factual
    city-attribute assertion.
    """

    if isinstance(clue, CitySceneValue):
        item = clue
    else:
        item = city_scene_value(clue[0], clue[1])
    if item is None:
        raise ValueError("unknown city scene clue")
    city = " ".join(str(city_name).split())
    country = " ".join(str(country_name).split())
    if not city or not country:
        raise ValueError("city and country names must be non-empty")
    return tuple(
        template.format(
            city=city,
            country=country,
            description=item.description,
        )
        for template in CITY_SCENE_PROMPT_TEMPLATES
    )


# Import-time structural invariants keep the fixed prompt space bounded.
if tuple(CITY_SCENE_VALUES) != CITY_SCENE_DIMENSIONS:
    raise RuntimeError("city scene dimensions and value groups differ")
if len(CITY_SCENE_VALUE_BY_DIMENSION_VALUE) != sum(
    len(values) for values in CITY_SCENE_VALUES.values()
):
    raise RuntimeError("duplicate canonical city scene dimension/value")
if len(CITY_SCENE_PROMPT_TEMPLATES) != 2:
    raise RuntimeError("city scene prompt template count must be two")
if any(
    not item.label or not item.description
    for values in CITY_SCENE_VALUES.values()
    for item in values
):
    raise RuntimeError("city scene values require fixed labels and descriptions")


__all__ = [
    "CITY_SCENE_DIMENSIONS",
    "CITY_SCENE_PROMPT_TEMPLATES",
    "CITY_SCENE_VALUE_BY_DIMENSION_VALUE",
    "CITY_SCENE_VALUE_DESCRIPTIONS",
    "CITY_SCENE_VALUE_LABELS",
    "CITY_SCENE_VALUES",
    "CITY_SCENE_VERSION",
    "MAX_CITY_SCENE_CITIES",
    "MAX_CITY_SCENE_CLUES",
    "MAX_CITY_SCENE_PROMPTS",
    "CitySceneValue",
    "city_scene_value",
    "normalize_city_scene_clue",
    "render_city_scene_prompts",
    "validate_city_scene_clues",
]

