-- Normalized base query for malaria objects.
-- Placeholders:
--   base_query.sql  -> contents of base_query.sql (a SELECT without trailing ';').
--
-- After substitution this expands to a SELECT that:
--   - returns all columns from base_query
--   - adds normalized stage_n and species_n fields.

SELECT
    b.*,
    REGEXP_REPLACE(LOWER(COALESCE(b.stage,   '')), '[^a-z]', '') AS stage_n,
    REGEXP_REPLACE(LOWER(COALESCE(b.species, '')), '[^a-z]', '') AS species_n
FROM (
    {{base_query}}
) AS b

