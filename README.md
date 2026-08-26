# #NisamPrijavila (#IDidntReport): reproducibility package

Supporting materials for "From solidarity to broadcast: Network dynamics of feminist hashtag activism in #NisamPrijavila (#IDidntReport)".

## Package contents

    data/deidentified_events.csv              25,117 de-identified event records
    data/deidentified_events.schema.md        field definitions and privacy limits
    data/deidentified_events.manifest.json    input/output hashes and count checks
    data/motive_codebook.csv                  manual motive-code definitions
    code/                                     privacy-screened analysis and release code
    derived_outputs/                          privacy-screened aggregate tables

The event table supports analysis of the paper phases, event types, reply and retweet relations, emotion annotations, institution labels, and reported manual annotations. Multi-label fields are compact JSON arrays.

## Privacy scope

The package excludes post text, original X event IDs, original account IDs, account names, profile fields, URLs, annotator identifiers, and crosswalks between source and release tokens. Event and account tokens are fresh random UUIDv4 values generated for this release. Timestamps have minute precision and receive random integer jitter of up to 15 minutes while preserving the original UTC date and paper phase.

The data are de-identified and pseudonymous. Timing, network structure, and rare annotation combinations retain information and may permit linkage to outside sources. The package therefore makes no claim of full anonymity. Attempts to identify accounts or individual disclosures are outside the intended use of the release.

The ZIP builder screens derived tables and code for restricted columns and compares every staged text file with source event IDs, account IDs, and handles. The build stops if a source identity survives the screen. Binary figures are excluded because they cannot receive the same exact token-level audit.

## Archive counts

The analytic archive contains 25,117 unique events from 8,637 observed author account-handle pairs: 20,427 retweets, 3,874 original posts, and 816 replies. Six events fall outside the paper's primary phase window and are labeled explicitly.

## Reproducibility boundary

The release reproduces analyses that use the provided de-identified fields and aggregate outputs. Collection, text annotation, identity resolution, and adjudication require restricted source data and cannot be rerun from this package. `REPRODUCE.md` describes the checks that can be run on the public artifact.
