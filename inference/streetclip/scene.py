"""Fixed, bounded vocabulary for zero-shot StreetCLIP scene analysis.

Only prompts defined here are ever encoded.  The endpoint accepts an image,
never caller-supplied prompt text, so the response vocabulary cannot be
arbitrarily expanded per request.  Each option carries a small number of
English street-scene prompts plus a fixed Chinese display label.

These labels are semantic hypotheses from image/text cosine comparison.  They
are not object detections, calibrated probabilities or geographic evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

SCENE_PROMPT_TEMPLATE_VERSION = "streetclip-scene-prompts-v1"

# Deliberately conservative and intentionally uncalibrated.  A group whose two
# best cosine scores are closer than this is flagged uncertain for review.
SCENE_UNCERTAIN_MARGIN_THRESHOLD = 0.05
SCENE_MAX_CANDIDATES = 2


@dataclass(frozen=True, slots=True)
class SceneOption:
    """One fixed scene concept and its curated prompt variants."""

    value: str
    label: str
    prompts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SceneGroupDefinition:
    """One fixed dimension and its bounded option set."""

    dimension: str
    options: tuple[SceneOption, ...]


SCENE_GROUPS: tuple[SceneGroupDefinition, ...] = (
    SceneGroupDefinition(
        dimension="road",
        options=(
            SceneOption(
                value="arterial_road",
                label="主干道",
                prompts=(
                    "a street-level photo of a wide urban arterial road with multiple lanes",
                    "a city road with heavy traffic and multiple vehicle lanes",
                ),
            ),
            SceneOption(
                value="local_street",
                label="城市街道",
                prompts=(
                    "a street-level photo of a narrow city street between buildings",
                    "a local street in a built-up urban area",
                ),
            ),
            SceneOption(
                value="residential_lane",
                label="住宅道路",
                prompts=(
                    "a residential lane between homes",
                    "a quiet street in a residential neighborhood",
                ),
            ),
            SceneOption(
                value="highway",
                label="高速公路",
                prompts=(
                    "a highway or motorway photographed from the roadside",
                    "a multi-lane high-speed road with grade separation",
                ),
            ),
            SceneOption(
                value="unpaved_road",
                label="土路或碎石路",
                prompts=(
                    "an unpaved dirt or gravel road",
                    "a rural road without paved asphalt",
                ),
            ),
            SceneOption(
                value="pedestrian_way",
                label="步行道",
                prompts=(
                    "a paved pedestrian sidewalk or walkway",
                    "a pedestrian-only path or plaza",
                ),
            ),
        ),
    ),
    SceneGroupDefinition(
        dimension="building",
        options=(
            SceneOption(
                value="high_rise",
                label="高层建筑",
                prompts=(
                    "a street scene dominated by tall high-rise towers",
                    "many-story apartment or office towers seen from street level",
                ),
            ),
            SceneOption(
                value="mid_rise",
                label="中层建筑",
                prompts=(
                    "a street lined with mid-rise apartment or office buildings",
                    "four to ten story urban buildings along a street",
                ),
            ),
            SceneOption(
                value="low_rise",
                label="低层建筑",
                prompts=(
                    "a street lined with low-rise houses",
                    "single-family or small residential buildings along a street",
                ),
            ),
            SceneOption(
                value="historic_masonry",
                label="历史砖石建筑",
                prompts=(
                    "a street with historic brick or stone buildings",
                    "older masonry architecture along a street",
                ),
            ),
            SceneOption(
                value="commercial_storefront",
                label="商业店面",
                prompts=(
                    "a street with ground-floor shops and storefronts",
                    "a commercial retail street with signs and shop windows",
                ),
            ),
            SceneOption(
                value="industrial_warehouse",
                label="工业或仓储建筑",
                prompts=(
                    "a street with warehouses or industrial buildings",
                    "an industrial district with large utilitarian structures",
                ),
            ),
        ),
    ),
    SceneGroupDefinition(
        dimension="vegetation",
        options=(
            SceneOption(
                value="street_trees",
                label="行道树",
                prompts=(
                    "street trees lining an urban road",
                    "a tree-lined sidewalk or avenue",
                ),
            ),
            SceneOption(
                value="dense_green",
                label="茂密绿地",
                prompts=(
                    "dense green vegetation surrounding a street",
                    "a street bordered by thick green foliage",
                ),
            ),
            SceneOption(
                value="sparse_green",
                label="稀疏植被",
                prompts=(
                    "only sparse vegetation around a street",
                    "a mostly paved street scene with little greenery",
                ),
            ),
            SceneOption(
                value="grass_lawn",
                label="草地或草坪",
                prompts=(
                    "a street beside a grassy lawn or field",
                    "a mowed grass area next to a road",
                ),
            ),
            SceneOption(
                value="flowering_plants",
                label="花卉植物",
                prompts=(
                    "flowering plants or shrubs beside a street",
                    "visible flowers and garden vegetation near a road",
                ),
            ),
            SceneOption(
                value="dry_vegetation",
                label="干燥植被",
                prompts=(
                    "dry scrub or arid vegetation beside a road",
                    "brown dry vegetation in a street scene",
                ),
            ),
        ),
    ),
    SceneGroupDefinition(
        dimension="terrain",
        options=(
            SceneOption(
                value="flat_ground",
                label="平坦地面",
                prompts=(
                    "a street on visibly flat level ground",
                    "a level street with no apparent slope",
                ),
            ),
            SceneOption(
                value="sloped_ground",
                label="坡地",
                prompts=(
                    "a street on a noticeable slope or incline",
                    "a road climbing or descending a hillside",
                ),
            ),
            SceneOption(
                value="hilly",
                label="丘陵",
                prompts=(
                    "a street scene among rolling hills",
                    "hilly terrain with roads following the hills",
                ),
            ),
            SceneOption(
                value="mountainous",
                label="山地",
                prompts=(
                    "a street in a mountainous landscape",
                    "steep mountain terrain visible around a road",
                ),
            ),
            SceneOption(
                value="valley",
                label="谷地",
                prompts=(
                    "a street in a valley between elevated terrain",
                    "a road in a low valley with slopes on both sides",
                ),
            ),
            SceneOption(
                value="coastal_lowland",
                label="沿海低地",
                prompts=(
                    "a street on low coastal terrain",
                    "a road near the coast at low elevation",
                ),
            ),
        ),
    ),
    SceneGroupDefinition(
        dimension="water",
        options=(
            SceneOption(
                value="no_visible_water",
                label="未见水体",
                prompts=(
                    "a street scene with no visible body of water",
                    "a dry street view without rivers lakes or sea",
                ),
            ),
            SceneOption(
                value="river",
                label="河流",
                prompts=(
                    "a street beside a river",
                    "a river visible from a road or bridge",
                ),
            ),
            SceneOption(
                value="lake",
                label="湖泊",
                prompts=(
                    "a street beside a lake",
                    "a lake visible from a road or shoreline",
                ),
            ),
            SceneOption(
                value="sea_ocean",
                label="海洋",
                prompts=(
                    "a street beside the sea or ocean",
                    "coastal water visible from a road or promenade",
                ),
            ),
            SceneOption(
                value="canal",
                label="运河",
                prompts=(
                    "a street beside a canal",
                    "an urban canal visible from a road",
                ),
            ),
            SceneOption(
                value="pond_fountain",
                label="水池或喷泉",
                prompts=(
                    "a small pond or fountain visible in a street scene",
                    "a decorative water feature beside a street",
                ),
            ),
        ),
    ),
    SceneGroupDefinition(
        dimension="vehicle",
        options=(
            SceneOption(
                value="cars",
                label="汽车",
                prompts=(
                    "a street scene with passenger cars",
                    "multiple automobiles on a road",
                ),
            ),
            SceneOption(
                value="motorcycles_scooters",
                label="摩托车或踏板车",
                prompts=(
                    "motorcycles or scooters on a street",
                    "a street scene with a motorbike or scooter",
                ),
            ),
            SceneOption(
                value="bicycles",
                label="自行车",
                prompts=(
                    "bicycles on or beside a street",
                    "a street scene with a bicycle",
                ),
            ),
            SceneOption(
                value="buses",
                label="公交车",
                prompts=(
                    "a bus on a city street",
                    "public buses visible on a road",
                ),
            ),
            SceneOption(
                value="trucks",
                label="卡车",
                prompts=(
                    "trucks on a street or road",
                    "a street scene with a delivery truck or lorry",
                ),
            ),
            SceneOption(
                value="no_visible_vehicles",
                label="未见车辆",
                prompts=(
                    "a street scene with no visible vehicles",
                    "an empty road or pedestrian area without cars buses or trucks",
                ),
            ),
        ),
    ),
)

SCENE_DIMENSIONS: tuple[str, ...] = tuple(group.dimension for group in SCENE_GROUPS)
SCENE_GROUP_BY_DIMENSION: dict[str, SceneGroupDefinition] = {
    group.dimension: group for group in SCENE_GROUPS
}


def iter_scene_prompts() -> tuple[str, ...]:
    """Return prompts in a deterministic group/option/variant order."""
    return tuple(
        prompt
        for group in SCENE_GROUPS
        for option in group.options
        for prompt in option.prompts
    )


__all__ = [
    "SCENE_DIMENSIONS",
    "SCENE_GROUP_BY_DIMENSION",
    "SCENE_GROUPS",
    "SCENE_MAX_CANDIDATES",
    "SCENE_PROMPT_TEMPLATE_VERSION",
    "SCENE_UNCERTAIN_MARGIN_THRESHOLD",
    "SceneGroupDefinition",
    "SceneOption",
    "iter_scene_prompts",
]