SELECT
    -- Object location & metadata (from Query 1)
    o.id as object_id,
    o.horizontal_location as x,
    o.vertical_location as y,
    o.pixelIdxList,
    o.locator_value,
    o.location_date,

    -- Image & sample info
    ss.name as PID,
    ss.description as sample_desc,
    s.id as id_image_set,
    s.id_slide_number,
    s.z_stack_height,
    s.pixels_per_micron,

    -- Tile info
    t.id as id_image_tile,
    t.filename as tile_filename,
    t.file_location,
    t.z_stack_index as z_index,
    t.z_stage_position,
    t.annotated,

    -- Blood sample info
    bs.id as id_blood_sample,
    bs.collection_date,

    -- Locator info
    loc.id as id_locator_algorithm,
    loc.name as locator_algorithm,
    loc.description as locator_algorithm_desc,

    -- Scanner info
    icc.id as id_icc,
    mo.magnification,

    -- Smear type
    sm.name as smear_type,

    -- PARASITEMIA (from Query 2 - external_analyses)
    COALESCE(SUM(DISTINCT ea.unstaged_count_per_microliter), -1) as parasitemia,
    COALESCE(SUM(DISTINCT ea.pcr_parasitemia_per_microliter), -1) as pcr_parasitemia,
    COALESCE(SUM(DISTINCT ea.ring_count_per_microliter), 0) as ring_count_per_microliter,
    COALESCE(SUM(DISTINCT ea.trophozoite_count_per_microliter), 0) as trophozoite_count_per_microliter,
    COALESCE(SUM(DISTINCT ea.gametocyte_count_per_microliter), 0) as gametocyte_count_per_microliter,
    COALESCE(SUM(DISTINCT ea.schizont_count_per_microliter), 0) as schizont_count_per_microliter,

    -- OBJECT LABELS (from Query 3 - object_analyses)
    -- May be NULL if object doesn't have manual labels
    MAX(CASE WHEN fld.field_name = 'stage' THEN re.name END) as stage,
    MAX(CASE WHEN fld.field_name = 'species' THEN re.name END) as species,
    COALESCE(MAX(CASE WHEN fld.field_name = 'in_focus' THEN anal.analysis_score END), 0) as in_focus,

    -- COMPUTED FIELDS
    REGEXP_REPLACE(t.filename, '_z[0-9]+\\.([a-z]+)$', '') as z_stack_filename

FROM objects_of_interest o

-- Base joins (from original Query 1)
JOIN locator_algorithms loc ON o.id_locator_algorithm = loc.id
JOIN image_tiles t ON o.id_image_tile = t.id
JOIN image_sets s ON t.id_image_set = s.id
JOIN smear_types sm ON s.id_smear_type = sm.id
JOIN blood_samples bs ON s.id_blood_sample = bs.id
JOIN sample_sets ss ON bs.id_sample_set = ss.id
JOIN image_capture_configurations icc ON s.id_image_capture_configuration = icc.id
JOIN objectives mo ON icc.id_objective = mo.id

-- ADD: Parasitemia joins (Query 2)
LEFT JOIN external_analyses ea ON ea.id_blood_sample = bs.id
    AND ea.id_malaria_species IN (
        SELECT id FROM malaria_species WHERE id != 6
    )

-- ADD: Object label joins (Query 3)
LEFT JOIN object_analyses anal ON anal.id_object = o.id
LEFT JOIN object_algorithm_fields fld ON anal.id_object_algorithm_field = fld.id
    AND fld.id_object_algorithm IN (3, 4)
LEFT JOIN object_result_types rt ON fld.id_object_result_type = rt.id
LEFT JOIN object_result_enumeration re ON rt.id_object_enumeration = re.id_object_enumeration
    AND anal.analysis_score_enumerated = re.result_value
LEFT JOIN object_analysis_files af ON anal.id_object_analysis_file = af.id

-- WHERE
--     loc.id IN (33, 35)
--     AND s.pixels_per_micron < 9.0
--     AND s.id IN (12545, 13880, 14082, 14084, 14100) -- image sets
--     AND sm.name = 'THICKSMEAR'
--     AND icc.id = 14

GROUP BY
    o.id,
    o.horizontal_location,
    o.vertical_location,
    o.pixelIdxList,
    o.locator_value,
    o.location_date,
    ss.name,
    ss.description,
    s.id,
    s.id_slide_number,
    s.z_stack_height,
    s.pixels_per_micron,
    t.id,
    t.filename,
    t.file_location,
    t.z_stack_index,
    t.z_stage_position,
    t.annotated,
    bs.id,
    bs.collection_date,
    loc.id,
    loc.name,
    loc.description,
    icc.id,
    mo.magnification,
    sm.name