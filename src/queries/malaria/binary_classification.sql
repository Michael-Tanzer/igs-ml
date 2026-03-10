-- binary_labels.sql
-- Pure "is this malaria?" binary classification labels for thicksmear.
--
-- Aligned with MATLAB pipeline:
--   S028_IsParasite_v2.m  — positive/negative/excluded logic
--   IG007_IgnoreDoubtfulLateAndRingIgnore.m — ignorable remapping
--   ExtractThumbnails.m   — tcIncludes=[1,3] default filter
--   collate_*.m           — species/stage definitions
--
-- Expects norm_wrapper providing: stage_n, species_n, in_focus, smear_type.
-- stage_n  = REGEXP_REPLACE(LOWER(COALESCE(stage,'')), '[^a-z]', '')
-- species_n = REGEXP_REPLACE(LOWER(COALESCE(species,'')), '[^a-z]', '')

SELECT
    n.*,

    /* ═══════════════════════════════════════════════════════════════════
     * true_class_approx
     * ─────────────────
     * Reconstructs MATLAB's true_class from DB annotation fields.
     * MATLAB assigns true_class in S042_DetectionThreePassZStackSuppress
     * based on in_focus x detected x suppressed flags. In the DB we only
     * have annotation (species, stage, in_focus), so this is approximate.
     *
     * MATLAB true_class values:
     *   1  = in-focus detected parasite
     *   2  = out-of-focus / doubtful parasite
     *   3  = distractor (false positive)
     *   4  = undetected in-focus parasite  (not in DB -- never detected)
     *   5  = suppressed distractor         (not in DB -- suppressed)
     *   6  = suppressed parasite           (not in DB -- suppressed)
     *   7  = undetected out-of-focus       (not in DB -- undetected)
     *   8  = ignorable parasite (via IG007 remapping)
     *   9  = exclusion zone object         (not in DB -- spatial filter)
     *  10  = high-confidence WBC
     *  11  = suspicious WBC
     *  12  = P2->P3 annotation exclusion   (not in DB -- spatial filter)
     *  20  = RBC                           (not in DB -- separate pipeline)
     * ═══════════════════════════════════════════════════════════════════ */
    CASE
        /* IG007 lines 15-20: late stages -> true_class=8 (ignorable)
         * regardless of in_focus. Applied BEFORE focus-based classification. */
        WHEN n.stage_n IN (
            'troph','trophozoite','schizont','schiz',
            'gametocyte','gameto','late','latestage',
            'unsurelatestage','lateignore','latestageignore'
        ) THEN 8

        /* IG007: ring ignores -> true_class=8 (ignorable) */
        WHEN n.stage_n IN ('ringignore','unsureringtroph')
        THEN 8

        /* IG007: doubtful -> true_class=2 */
        WHEN n.stage_n = 'doubtful'
          OR n.stage_n LIKE '%doubt%'
        THEN 2

        /* WBC: species or stage indicates WBC */
        WHEN n.stage_n = 'wbc' OR n.species_n = 'wbc'
        THEN 10

        /* In-focus confirmed ring parasite -> true_class=1 */
        WHEN n.in_focus != 0
         AND n.stage_n IN ('pfring','pvring','poring','pmring','ring','ringnucleusonly')
        THEN 1

        /* Out-of-focus confirmed ring parasite -> true_class=2 */
        WHEN n.in_focus = 0
         AND n.stage_n IN ('pfring','pvring','poring','pmring','ring','ringnucleusonly')
        THEN 2

        /* Distractor: unknown species with "detected" stage,
         * or explicitly labeled distractor/other/none -> true_class=3 */
        WHEN n.stage_n IN ('distractor','other','none','')
        THEN 3
        WHEN n.species_n = 'unknown' AND n.stage_n = 'detected'
        THEN 3

        /* Non-parasite catch-all */
        WHEN n.stage_n IN ('duplicatetoignore','ideopathic','unvetted')
        THEN 3

        ELSE NULL
    END AS true_class_approx,


    /* ═══════════════════════════════════════════════════════════════════
     * y_binary_matlab
     * ───────────────
     * Literal MATLAB default CNN training:
     *   tcIncludes=[1,3], doubtful_flag=0, ignore_flag=0
     *
     * S028_IsParasite_v2 + IG007 + ExtractThumbnails:
     *   IG007 remaps late stages -> tc=8 (ignorable)
     *   IG007 remaps ring ignores -> tc=8
     *   IG007 remaps doubtful -> tc=2
     *   ExtractThumbnails keeps only tc IN (1,3)
     *   S028: tc=1 -> positive, tc=3 -> negative
     *
     * Net: only in-focus ring parasites as positive, distractors as
     * negative. Late stages, out-of-focus, doubtful all excluded.
     * ═══════════════════════════════════════════════════════════════════ */
    CASE
        /* Negative (~ tc=3): distractors and non-parasites */
        WHEN n.species_n = 'unknown' AND n.stage_n = 'detected'
        THEN 0
        WHEN n.stage_n IN ('distractor','other','none','')
        THEN 0
        WHEN n.stage_n IN ('duplicatetoignore','ideopathic')
        THEN 0

        /* Positive (~ tc=1): in-focus confirmed ring parasites only */
        WHEN n.stage_n IN ('pfring','pvring','poring','pmring','ring','ringnucleusonly')
        THEN 1
    
        /* Everything else excluded: out-of-focus (tc=2), late stages (tc=8),
         * doubtful (tc=2), ignore stages (tc=8), WBCs, unvetted, unsure */
        ELSE NULL
    END AS y_binary_matlab,


    /* ═══════════════════════════════════════════════════════════════════
     * y_binary_strict
     * ───────────────
     * Confirmed parasites only -- includes all stages but excludes
     * doubtful/unsure/ignore annotations. Middle ground between
     * y_binary_matlab (very narrow) and y_binary_inclusive (very broad).
     *
     * Positive: confirmed ring + late parasites (any focus)
     * Negative: distractors, WBCs, non-parasites
     * NULL: doubtful, unsure, ignore, unvetted
     * ═══════════════════════════════════════════════════════════════════ */
    CASE
        /* Negative: distractors and non-parasites */
        WHEN n.species_n = 'unknown' AND n.stage_n = 'detected'
        THEN 0
        WHEN n.stage_n IN ('distractor','wbc','other','none','')
        THEN 0
        WHEN n.species_n = 'wbc'
        THEN 0
        WHEN n.stage_n IN ('duplicatetoignore','ideopathic')
        THEN 0

        /* Positive: confirmed ring parasites (any focus) */
        WHEN n.stage_n IN (
            'pfring','pvring','poring','pmring','ring','ringnucleusonly'
        ) THEN 1

        /* Positive: confirmed late-stage parasites (any focus) */
        WHEN n.stage_n IN (
            'troph','trophozoite','schizont','schiz',
            'gametocyte','gameto','late','latestage'
        ) THEN 1

        /* Excluded: doubtful, unsure, ignore, unvetted */
        ELSE NULL
    END AS y_binary_strict,


    /* ═══════════════════════════════════════════════════════════════════
     * y_binary_inclusive
     * ─────────────────
     * Matches MATLAB with doubtful_flag=1, ignore_flag=1.
     *
     * Positive: all parasite stages (any focus, any confidence)
     * Negative: confirmed non-parasites
     * NULL: only truly unclassifiable
     * ═══════════════════════════════════════════════════════════════════ */
    CASE
        /* Negative: distractors and non-parasites */
        WHEN n.species_n = 'unknown' AND n.stage_n = 'detected'
        THEN 0
        WHEN n.stage_n IN ('distractor','wbc','other','none','')
        THEN 0
        WHEN n.species_n = 'wbc'
        THEN 0
        WHEN n.stage_n IN ('duplicatetoignore','ideopathic')
        THEN 0

        /* Positive: all confirmed parasite stages (any focus level) */
        WHEN n.stage_n IN (
            'pfring','pvring','poring','pmring','ring','ringnucleusonly',
            'troph','trophozoite','schizont','schiz',
            'gametocyte','gameto','late','latestage'
        ) THEN 1

        /* Positive: doubtful/unsure stages (included in this variant) */
        WHEN n.stage_n IN (
            'doubtful','unsure','unsurelate','unsurelatestage','unsureringtroph'
        )
        OR n.stage_n LIKE '%doubt%'
        OR n.stage_n LIKE '%unsure%'
        THEN 1

        /* Positive: *ignore stages are real parasites with low confidence */
        WHEN n.stage_n IN ('ringignore','lateignore','latestageignore')
        THEN 1

        /* Excluded: unvetted, empty, or truly unknown */
        ELSE NULL
    END AS y_binary_inclusive,


    /* ═══════════════════════════════════════════════════════════════════
     * label_category
     * ──────────────
     * Mirrors MATLAB IG007 + S028_IsParasite_v2 for per-node CNN training.
     * Downstream Python uses this to build stage-specific pos/neg sets:
     *   pfRing:  positive_ring(1) vs negative(0), exclude late_stage
     *   pvRing:  positive_ring(1) vs negative(0), exclude late_stage
     *   late:    late_stage(1) vs negative+positive_ring(0)
     *   ring:    positive_ring(1) vs negative(0)
     * ═══════════════════════════════════════════════════════════════════ */
    CASE
        /* 1. Negative: annotated non-parasites */
        WHEN n.species_n = 'unknown' AND n.stage_n = 'detected'
        THEN 'negative'
        WHEN n.stage_n IN ('distractor','wbc','other','none','')
        THEN 'negative'
        WHEN n.species_n = 'wbc'
        THEN 'negative'

        /* 2. Late stages -- IG007 remaps to ignorable (tc=8).
         *    Positive for late CNN, excluded for ring CNN. */
        WHEN n.stage_n IN (
            'troph','trophozoite','schizont','schiz',
            'gametocyte','gameto','late','latestage',
            'unsurelatestage','lateignore','latestageignore'
        ) THEN 'late_stage'

        /* 3. Ring ignores -- IG007 remaps to ignorable (tc=8) */
        WHEN n.stage_n IN ('ringignore','unsureringtroph')
        THEN 'ring_ignore'

        /* 4. Doubtful/unvetted -- IG007 remaps to tc=2 */
        WHEN n.stage_n IN ('doubtful','unvetted')
          OR n.stage_n LIKE '%doubt%'
          OR n.stage_n LIKE '%unsure%'
        THEN 'doubtful'

        /* 5. In-focus confirmed ring parasites (tc=1) */
        WHEN n.in_focus != 0
         AND n.stage_n IN ('pfring','pvring','poring','pmring','ring','ringnucleusonly')
        THEN 'positive_ring'

        /* 6. Out-of-focus confirmed ring parasites (tc=2) */
        WHEN n.in_focus = 0
         AND n.stage_n IN ('pfring','pvring','poring','pmring','ring','ringnucleusonly')
        THEN 'out_of_focus'

        /* 7. Non-parasite catch-all */
        WHEN n.stage_n IN ('duplicatetoignore','ideopathic')
        THEN 'negative'

        ELSE NULL
    END AS label_category

FROM (
    {{norm_wrapper}}
) AS n
WHERE n.smear_type = 'THICKSMEAR';
