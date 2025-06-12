
### 

Expand conversions

```sql
SELECT 
	schema_conversions.*,
	input_chunks.content as input_content,
	output_chunks.content as output_content,
	json_schemas.content as schema_content
FROM schema_conversions
JOIN chunks AS input_chunks ON input_chunk_id = input_chunks.id
JOIN chunks AS output_chunks ON output_chunk_id = output_chunks.id
JOIN json_schemas ON schema_conversions.schema_id = json_schemas.id
```