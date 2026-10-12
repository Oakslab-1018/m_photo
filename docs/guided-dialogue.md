# Guided dialogue implementation contract

This adds a persisted country -> admin1 -> city flow. The existing upload,
feedback and read-only city APIs retain their legacy contracts. The new frontend
starts guided mode after upload; model ranks are evidence, never confirmations.

## HTTP

- POST /api/geo/{session_id}/dialogue/start: initialize once; repeated calls return
  existing state without resetting selections or adding duplicate messages.
- GET /api/geo/{session_id}/dialogue: current snapshot; 404 if not initialized.
- POST /api/geo/{session_id}/dialogue/events: validate and apply one event atomically.

All return `{dialogue, candidates, action, messages, scores, clues, exclusions}`. An optional `inference_notice` explains unavailable model results or insufficient reliable evidence; it is shown even before candidate selection.
`action` retains existing kind/message/payload shape. `scores`, `clues`, and
`exclusions` retain legacy shapes where relevant; guided review lives in dialogue.

## Types

LocationTarget:
`{level: country|admin1|city, id: string, label: string, country_iso: string,
source_iso?: string, admin1_id?: string, city_id?: string, lat?: number, lng?: number}`.
Country IDs are canonical ISO; admin1 IDs are raw source ISO + '.' + string code;
city IDs are decimal strings. Validate each ID and parent against local catalogs.
Targets may also carry coordinate_source, coordinate_quality and coordinate_city_id.
Directory identities and scoped admin1/city reads use GeoCatalogStore;
admin1 coordinates use the documented derived representative point when available.
The shipped SQLite seed initializes fresh runtime databases. There is no directory JSON fallback.
Source scopes TW/HK/MO follow existing canonical CN policy; retain source metadata.

Candidate: `{target: LocationTarget, score?: number, score_type?: string,
reason?: string}`. Do not invent probability/model scores for catalog-only rows.

Dialogue:
`{version: 1, revision: number, stage: country|admin1|city|resolved,
confirmed: {country: LocationTarget|null, admin1: LocationTarget|null,
city: LocationTarget|null}, exclusions: LocationTarget[],
clues: ReviewedClue[], pending_question: GuidedQuestion|null}`.
History/idempotency are persisted internally. Every effective event increments
revision. Repeated event_id with the same semantic payload returns its original receipt
without reapplying; reusing an ID with a different payload returns 409. Revision
changes also cover evidence updated by point analysis or other session operations.

ReviewedClue:
`{id: string, dimension: string, value: string, description: string,
source: string, confidence?: number, status: pending|confirmed|rejected|uncertain}`.
Rejected evidence is excluded from guided reasoning but retained for audit.
Confirming an observation never confirms a geographic location. Preserve original
machine evidence and feed only appropriate catalog-backed evidence into ranking.

GuidedQuestion:
`{id: string, kind: review_clue|ask|confirm_location|request_clue,
message: string, dimension?: string, clue_id?: string,
options?: {value: string, label: string}[]}`.
Questions are scoped to the current stage and have stable IDs until answered.
A supplied question_id must match the current question kind and its clue/target.
Free-text negatives and corrections create an explicit ask with dimension
correction and apply/keep choices; they are not silently discarded. Applying a
validated correction can change evidence/exclusions or move backward, but never
confirms a location or advances to a deeper stage.
`unknown` means unavailable evidence, never rejection. Move to another question,
without inventing evidence or changing stage. Ask one principal question per turn.

Event:
`{type: confirm_clue|reject_clue|confirm_location|reject_location|answer|unknown|
retract|backtrack|free_text, event_id: string, expected_revision: number,
question_id?: string, target?: {level: country|admin1|city,id: string},
clue_id?: string, value?: string, text?: string,
level?: country|admin1|city}`.
Answers and unknown events must reference the pending question. Enforce revision
with an atomic conditional write; stale/conflicting revisions/questions return
409 and do not mutate. Unknown catalog IDs, invalid ancestry/events return 422.
Responses may include `receipt: {event_id,status,summary}`.

## Transitions and exclusions

Country confirmation -> admin1; admin1 confirmation -> city; city confirmation ->
resolved. Only explicit confirm_location advances. Rankings and free text do not.
A confirmed path must have catalog-valid ancestry. Confirmation of excluded
locations requires explicit retraction first. Exclusion operates within its scope;
it must not eliminate similarly named locations elsewhere.
Backtrack(country) clears all confirmations; backtrack(admin1) clears admin1/city;
backtrack(city) clears city. Evidence and scoped exclusion history survive.
Retract(target) removes that exclusion; retract(clue_id) restores pending review.
Any operation invalidating a parent must clear child confirmations.

## Candidate and question policy

Reuse country reasoning/model prior. Admin1 options come from the authoritative
admin1 directory, with human-readable names, independent of the national top-10
city results. City candidates/search must be scoped to selected admin1 BEFORE any
limit. Provide catalog search via free text so cities omitted by model top-10 can
still be found. Missing catalog/model coverage is explicit, never fabricated.
Use deterministic clue verification and discriminating questions first. Text is
parsed into proposed evidence/targets only. Conflicts with the confirmed path are
reported for clarification without silently changing the path. No remote LLM call
is required to complete the flow. Coordinates are representative points, not GPS.

## Frontend

Display the three-step confirmed path; current candidates with confirm/exclude;
clue review with confirm/reject; one question with answer/unknown; exclusions with
restore; backtrack controls; explicit resolved city result. Use a separate guided
composable/component; preserve existing candidate city browsing as optional.
The backend remains authoritative. On 409 reload state and ask user to retry;
do not replay stale confirmation automatically. A question whose ID/kind/target
changed cannot be retried as the previous answer. Point analysis disables guided
actions until the authoritative evidence/revision snapshot has been reloaded.
New uploads clear guided state;
retry preserves event_id. Disable duplicate submission while applying an event.

## Validation

Cover transitions, clue/location distinction, unknown, parent validation, scoped
exclusion/restore, backtracking, free-text proposal, duplicate/stale events,
restart persistence, rollback, no-provider operation, and legacy API compatibility.

## Guided panel presentation

The primary panel shows one authoritative observation question at a time. The
left clue rail retains the full observation list. Completing the observations
reveals an explicit Next button; only that action opens the current level's
Top-5 candidates. Neither the button nor a ranking confirms a location.

Returning to the previous observation retracts its review through the server,
which makes that observation the current question again. Returning to a previous
location level uses backtrack and preserves evidence and exclusion history.
New pending observations or unresolved correction questions return the panel to
review. Client review history records successful actions within the current
page session and does not invent server state after a reload.