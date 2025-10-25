DELETE FROM chunks 
WHERE content_type = 'SQL' 
AND json_extract(meta, '$.converter') = 'ToSQLInsertConverter';

