SELECT schema_conversions.id,
	json_schemas.id as schema_id,
	input_chunk.content as input_chunk_content, 
	json_schemas.content as schema_content,
	output_chunk.content as output_chunk_content
FROM schema_conversions
JOIN chunks as input_chunk ON schema_conversions.input_chunk_id = input_chunk.id
JOIN chunks as output_chunk ON schema_conversions.output_chunk_id = output_chunk.id
JOIN json_schemas ON schema_conversions.schema_id = json_schemas.id
WHERE schema_conversions.meta -> 'group' = "0"

SELECT schema_conversions.meta -> 'group' , COUNT( *) as group_size
FROM schema_conversions 
GROUP BY schema_conversions.meta -> 'group' 
ORDER BY group_size DESC
-- WHERE schema_conversions.meta -> 'group' = "2"