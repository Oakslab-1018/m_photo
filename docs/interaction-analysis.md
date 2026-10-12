# Guided map and local-region analysis contract

This extends the existing application without changing the legacy country/city
catalog or guided-event semantics. Do not run tests/builds for this task.

## Map

Guided mode reads LocationTarget candidates and confirmed country/admin1/city.
Selecting a target highlights it and focuses its representative point without
confirming it. Explicit confirmation advances the state and focuses the next
scope. Backtrack/exclusion removes stale selection and focuses the returned scope.
Country/admin1/city/resolved camera zooms are roughly 4/7/10/12. Current-level
candidates determine bounds; confirmed parents may be shown but must not widen
those bounds. Missing coordinates use the closest confirmed-parent focus and a
visible missing-point explanation, never a fabricated coordinate or boundary.

## Local request

POST /api/geo/{id}/analyze-point keeps x/y normalized and adds optional:
- scale: float 0.08..0.8, default 0.25, relative to the shorter image side.
- image_region: {x,y,w,h}, normalized top-left rectangle inside the oriented image.
- purpose: auto|ocr|scene, default auto.
- client_request_id: optional stable request identifier for retry.

Use one EXIF-transposed RGB image for all coordinates. A click chooses a detail
crop; an explicit rectangle is authoritative. Auto/scene can additionally use a
larger context crop (about 0.6 of short side, containing the detail rectangle).
OCR can enlarge a small detail crop, but full-image boxes must account for both
scale and offset. Keep old crop_info x/y/size fields.

## Local response

Keep existing AnalyzePointResponse fields; add:
- focus_region: RegionReport
- model_status, ocr_status (may be aliases of report fields)
- applied: bool (false for duplicate report/no new evidence)

RegionReport:
{id, purpose, scale, bbox_px:[left,top,right,bottom],
 bbox_norm:{x,y,w,h}, context_bbox_norm?:{x,y,w,h},
 image:{width,height,orientation_applied,detail_width,detail_height,ocr_scale},
 quality:{issues:string[],metrics:object},
 ocr:{provider,enabled,available,state,line_count},
 model:{provider,enabled,available,state,ranking_applied,margin?,uncertain?},
 observations:RegionObservation[], question_suggestions:RegionQuestion[],
 summary:string, dedupe:new|duplicate_region|duplicate_request|no_new_evidence}

Observation:
{id,kind:ocr_text|semantic|pixel_signal|scene_hypothesis|crop_ranking,
 dimension,value,text?,label,evidence,source,provider?,bbox_norm?,quad_norm?,
 trust:{level:high|medium|low,reliable,validated,affects_ranking,
 confidence_kind?,confidence?,raw_cosine?,uncertain?}}

Questions: {id,text,reason,basis_observation_ids:string[]}.

CropInfo may add region_id, bbox_px,bbox_norm,scale,purpose. Reports and original
input are persisted in the AI ChatMessage.meta.point_analysis JSON (no schema
migration); region_id is included in new clue.extra and OCR provenance. Read
history from returned messages. Same input/region should not duplicate clues or
AI analysis reports; repeat request returns existing report plus current state.

## Grounding

Mock/unavailable OCR emits no real text observation. Low-score OCR is not marked
reliable. Invalid/zero-area OCR boxes are absent, not fabricated. Record actual
recognized text, recognition score and image-global boxes. Distinguish inferred
language/place semantic clues from recognized literal text.

Color-based heuristics must not assert gothic architecture, a vehicle, climate,
forest type or terrain. Show measured image signals and their limitations only.
Scene hypotheses come only from the optional real StreetCLIP scene endpoint and
are explicitly low-trust/review-only; raw cosine is not a location probability.
Questions are based on OCR/catalog matches/model ambiguity/available evidence.
No remote visual LLM or new model download is needed.

Optional scene endpoint POST /analyze-region on StreetCLIP receives same file+auth
as predict. Response {model_revision,image_sha256,status:ok,
 scene_groups:[{dimension,candidates:[{value,label,raw_cosine}],margin,uncertain}]}.
Groups are bounded curated road/building/vegetation/terrain/water/vehicle concepts;
max2 candidates per group, raw cosines finite in[-1,1]. Use shared encoder and
cached normalized text embeddings, single inference lock, existing weights.
Backend describe_region uses optional endpoint, validates against known groups,
returns unavailable explicitly on errors; scene hypotheses never affect ranking.

## Crop ranking and state

Only meaningful context crops may contribute weakly to country prior. Skip
uncertain, very small or poor-quality crops; retain diagnostic reasons. Preserve
all-image prior. Optional crop metadata region_id/purpose/bbox_norm/ranking_applied
must survive gateway validation; model fusion respects ranking_applied false and
groups repeated crops by region. Original legacy crops retain existing behavior.
Heavy model/OCR work happens before DB writes, then prepare_update and transaction
use identical evidence. Added evidence never confirms or changes geographic path;
frontend reloads authoritative guided state before reenabling controls.

## Local frontend

ImageViewer offers point or rectangle selection, scale controls, purpose
(auto/text/scene), visible ROI and explicit Analyze action. Show per-region report
with crop preview, actual text/boxes, provider availability, quality issues,
reference-only model hypotheses and suggested next questions. Preserve history
and errors, clear on new session. Reject stale session/request results, keep
client_request_id on retries, and disable guided/map mutations during analysis.
