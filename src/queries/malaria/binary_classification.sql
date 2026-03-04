-- Template for binary classification labels built on the normalized base query.
-- This file expects ``norm_wrapper.sql`` in the same directory and will usually
-- be expanded via ``render_sql`` before execution.

SELECT
    n.*,

    /* Strict: only clear parasite stages are positive */
    CASE
        WHEN n.stage_n IN (
            'distractor','wbc','other','unknown','',
            'none','unvetted','duplicatetoignore','ideopathic'
        )
        OR n.stage_n LIKE '%ignore%'
        THEN 0
        WHEN n.stage_n IN (
            'pfring','pvring','poring','pmring',
            'ring','troph','trophozoite','schizont','schiz',
            'gametocyte','gameto',
            'late','latestage',
            'ringnucleusonly'
        ) THEN 1
        ELSE NULL
    END AS y_binary_strict,

    /* Inclusive: doubtful/unsure also positive */
    CASE
        WHEN n.stage_n IN (
            'distractor','wbc','other','unknown','',
            'none','unvetted','duplicatetoignore','ideopathic'
        )
        OR n.stage_n LIKE '%ignore%'
        THEN 0
        WHEN n.stage_n IN (
            'pfring','pvring','poring','pmring',
            'ring','troph','trophozoite','schizont','schiz',
            'gametocyte','gameto',
            'late','latestage',
            'ringnucleusonly',
            'doubtful','unsure','unsurelate','unsurelatestage','unsureringtroph'
        )
        OR n.stage_n LIKE '%doubt%'
        OR n.stage_n LIKE '%unsure%'
        THEN 1
        ELSE NULL
    END AS y_binary_inclusive

FROM (
    {{norm_wrapper}}
) AS n;

