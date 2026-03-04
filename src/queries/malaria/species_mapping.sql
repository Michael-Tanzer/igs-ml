-- Template for species + stage-group labels built on the normalized base query.
-- This file expects ``norm_wrapper.sql`` in the same directory and will usually
-- be expanded via ``render_sql`` before execution.

SELECT
    p.*,

    /* 13-class mapping used by A021_speciesID_xgboost_predict */
    CASE
        WHEN p.stage_grp = 'distractor' THEN 0
        WHEN p.sp = 'pf' AND p.stage_grp = 'ring'     THEN 1
        WHEN p.sp = 'pv' AND p.stage_grp = 'ring'     THEN 2
        WHEN p.sp = 'po' AND p.stage_grp = 'ring'     THEN 3
        WHEN p.sp = 'pm' AND p.stage_grp = 'ring'     THEN 4
        WHEN p.sp = 'pf' AND p.stage_grp = 'unsurert' THEN 5
        WHEN p.sp = 'pv' AND p.stage_grp = 'unsurert' THEN 6
        WHEN p.sp = 'po' AND p.stage_grp = 'unsurert' THEN 7
        WHEN p.sp = 'pm' AND p.stage_grp = 'unsurert' THEN 8
        WHEN p.sp = 'pf' AND p.stage_grp = 'late'     THEN 9
        WHEN p.sp = 'pv' AND p.stage_grp = 'late'     THEN 10
        WHEN p.sp = 'po' AND p.stage_grp = 'late'     THEN 11
        WHEN p.sp = 'pm' AND p.stage_grp = 'late'     THEN 12
        ELSE NULL
    END AS y_species_13

FROM (
    SELECT
        n.*,

        /* Canonical species code: pf/pv/po/pm */
        CASE
            WHEN n.species_n IN ('pf','falc','falciparum') THEN 'pf'
            WHEN n.species_n IN ('pv','vivax')             THEN 'pv'
            WHEN n.species_n IN ('po','ovale')             THEN 'po'
            WHEN n.species_n IN ('pm','malariae')          THEN 'pm'
            WHEN n.stage_n LIKE 'pf%' THEN 'pf'
            WHEN n.stage_n LIKE 'pv%' THEN 'pv'
            WHEN n.stage_n LIKE 'po%' THEN 'po'
            WHEN n.stage_n LIKE 'pm%' THEN 'pm'
            ELSE NULL
        END AS sp,

        /* Stage group for species branch: ring / unsureRT / late / distractor */
        CASE
            WHEN n.stage_n LIKE '%distractor%' OR n.stage_n LIKE '%wbc%' THEN 'distractor'
            WHEN n.stage_n LIKE '%unsure%'                                THEN 'unsurert'
            WHEN n.stage_n LIKE '%ring%'                                  THEN 'ring'
            WHEN n.stage_n LIKE '%late%' OR n.stage_n LIKE '%troph%'
              OR n.stage_n LIKE '%schiz%' OR n.stage_n LIKE '%gamet%'     THEN 'late'
            ELSE NULL
        END AS stage_grp
    FROM (
        {{norm_wrapper}}
    ) AS n
) AS p;

