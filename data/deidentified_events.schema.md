# De-identified event table schema

`deidentified_events.csv` contains one row for each of the 25,117 unique events in the analytic archive. Empty scalar cells mean that an annotation or relationship is unavailable. Multi-label fields always contain a valid JSON array and use `[]` when no labels are available.

| Column | Type | Definition |
|---|---|---|
| `id` | string | Fresh random event token in the form `e_<uuid4 hex>`. It has no mathematical relation to the original X identifier. |
| `timestamp_utc` | UTC minute | Original event time with seconds removed and a random integer offset from -15 to +15 minutes. The transformation preserves the original UTC date and paper phase. |
| `anon_user_id` | string | Fresh random account token in the form `u_<uuid4 hex>`. A token represents one observed numeric-ID/handle pair within this release. |
| `phase` | category | `phase_1` through `phase_6`, following the paper boundaries, or `outside_primary_phase_window` for six collected events outside those boundaries. |
| `post_type` | category | `post`, `retweet`, or `reply`. Quote-post relationships were unavailable in the source archive and are not inferred. |
| `target_event_id` | string | Random event token for the retweeted or replied-to event. Empty for posts and for 46 retweets whose source event could not be resolved. Reply targets outside the collected archive receive opaque tokens so the relation can be represented without exposing the platform ID. |
| `target_anon_user_id` | string | Random account token for the retweeted or replied-to account. Empty for posts. |
| `emotion_gemma_4_31b` | category | Primary Gemma 4 31B emotion label reported in the paper. Retweets inherit the label of their source event. Eight unresolved retweets have no label. |
| `emotion_gpt_5` | category | GPT-5 emotion label for the reported 400-event validation sample. Empty outside that sample. |
| `emotion_gpt_5_4` | category | GPT-5.4 emotion label for the reported 400-event validation sample. Empty outside that sample. The unreported full-corpus GPT-5.4 run is excluded. |
| `human_emotion_labels` | JSON array | All student emotion-label votes for validation events, translated to the paper's English label set. Repeated labels represent independent votes. Annotator identifiers are excluded. |
| `human_emotion_modal` | category | Modal student label used as the human reference in the reported validation. |
| `institution_type_gpt5mini` | category | GPT-5-mini account classification: `personal`, `media`, `political`, `ngo`, or `unknown`. |
| `institution_type_final` | category | Final verified account classification: `personal`, `media`, `political`, or `ngo`. |
| `manual_reporting_status` | category | Manual reporting annotation: `reported`, `not_reported`, or `reported_and_not_reported`. Empty for events outside the 4,386 manually coded units or where the field was not assigned. |
| `manual_first_person` | Boolean string | Adjudicated first-person disclosure indicator (`true` or `false`) for the 4,386 manually coded units. |
| `manual_motive_codes` | JSON array | Parsed detailed manual motive codes, including special codes recorded during coding. Code definitions are in `motive_codebook.csv`. |
| `manual_motive_parent_categories` | JSON array | Parent motive categories used in the paper's analytic sample. Events marked `x` or `0`, and the unlisted `21` code, are excluded according to the published reconciliation rules. |

## Privacy transformation and limits

Event and account tokens are generated independently on each build. Their in-memory crosswalks are destroyed when the exporter exits and are never included in the package. The table contains no post text, original event or account IDs, account names, profile fields, URLs, annotator identifiers, or direct crosswalks.

This is de-identified, pseudonymous event-level research data. Exact anonymity cannot be guaranteed because timing, interaction structure, rare annotation combinations, and outside knowledge can support linkage attacks. Users should avoid attempts to re-identify accounts or disclosures and should follow the applicable ethics approval, data-use terms, and platform rules.

## Canonical counts

| Dimension | Count |
|---|---:|
| Events | 25,117 |
| Posts | 3,874 |
| Retweets | 20,427 |
| Replies | 816 |
| Unique author tokens | 8,637 |
| Retweets with target event | 20,381 |
| Replies with target event | 816 |
| Gemma labels | 25,109 |
| GPT-5 validation labels | 400 |
| GPT-5.4 validation labels | 400 |
| Events with human validation labels | 400 |
| Manually coded units | 4,386 |
| Events with any parsed motive code | 1,122 |
| Analytic motive events | 824 |
| Analytic detailed motive assignments | 1,266 |
| Analytic parent-category assignments | 1,238 |
