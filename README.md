# ANY2JSON

## Database 

This document outlines the **plan, ideas, and principles** for implementing a robust data storage solution for the `ANY2JSON` project. The primary goal is to replace the current reliance on JSON files with a more structured and maintainable SQLite database using SQLAlchemy ORM. This approach is designed to provide:

*   **Reproducibility**: The database will be a single file stored alongside the code, allowing for complete reproduction of the dataset construction.
*   **Traceability**: All data entities will be linked, enabling clear tracing of how each training sample was obtained.
*   **Augmentation**: The system will support the creation and storage of new, synthetically manipulated data, enabling dataset expansion.
*   **Querying and Sampling**: The structured nature of the database will facilitate efficient querying and sampling to create new datasets for model training.
*   **Data Integrity**: Leveraging a relational database with foreign keys ensures data consistency and integrity.

### Core Principles for Data Storage:

*   **Dataset as a File**: The database will function as a self-contained dataset file, not a production database.
*   **Lineage Tracking**: Every piece of data, whether original or synthetically generated, can be traced back to its origin.
*   **SQLAlchemy ORM**: All database operations will be performed using SQLAlchemy's ORM, avoiding raw SQL for a more Pythonic and maintainable codebase.
✨

### 1. Database Model using SQLAlchemy ORM

The following SQLAlchemy ORM models define the database schema for the `ANY2JSON` project, located in `database/models.py`. These models establish the entities and relationships necessary for data generation, augmentation, and training sample creation.

*   **`SourceDocument`**: This model represents an original document from which data is extracted. 
*   **`JsonSchema`**: This model stores JSON schema definitions. It supports storing schemas obtained by modifying other schemas via `parent_schema_id`.
*   **`Chunk`**: This model represents a segment of structured content, either original or synthetic. It supports storing bits of data in JSON and other formats through the `type` field. It supports storing synthetic chunks obtained from other chunks through `parent_chunk_id`and `is_synthetic` attributes. In case of synthetic chunks, details of how this Chunk was obtained are to be put in the optional `meta` attribute. 
*   **`TrainingSample`**: This model defines a single training example for the LLM. It links an `input_chunk`, `target_schema`, and `output_chunk` using their respective IDs (`input_chunk_id`, `schema_id`, `output_chunk_id`). 


### 2. Database Workflow

Here is how you would interact with the database based on the models above.

#### A. Setup

A single module, e.g., `database/client.py`, handles database initialization and session management. 

#### B. Populating the Database

1.  **Add a Source Document**: Read a raw file (XML, HTML, etc.) and store its content in the `SourceDocument` table.
2.  **Create Initial Chunks**: Process the `SourceDocument` to extract or convert parts of it into JSON. For each part, create a `JsonSchema` and a `Chunk` (with `is_synthetic=False`).

#### C. Data Augmentation

This workflow allows for creating new, synthetic data from existing chunks, which is crucial for building a robust training set.

1.  **Select a base chunk**: Query for an existing `Chunk`.
2.  **Modify the Schema**: Create a new `JsonSchema` object, setting its `parent_schema_id` to the original schema's ID.
3.  **Modify the Chunk**: Alter the base chunk's JSON content to conform to the new schema.
4.  **Save as new**: Create a new `Chunk` for this modified content. Set `is_synthetic=True` and link it to the original by setting `parent_chunk_id`.

#### D. Creating Training Samples

A training sample is a triplet that defines a conversion task.

1.  **Define the Task**: Select an input `Chunk` (this can be an original or synthetic chunk).
2.  **Define the Target**: Select a `JsonSchema` that the input should be converted to.
3.  **Define the Output**: Select the corresponding output `Chunk` that represents the correct conversion.
4.  **Create the Link**: Create a `TrainingSample` record that links these three entities via their IDs.

#### E. Generating a Dataset for the Model

When you need to generate a dataset for training your LLM, you can simply query the `TrainingSample` table. Each row provides the IDs to fetch the input string (from `input_chunk.content`), the target schema (from `target_schema.content`), and the expected output JSON (from `output_chunk.content`). This gives you a clean and reproducible way to build datasets from the stored data.

## Links

- https://github.com/thunlp/SchemaReinforcementLearning
- https://github.com/guidance-ai/jsonschemabench?tab=readme-ov-file
- https://github.com/plenaryapp/awesome-rss-feeds?tab=readme-ov-file#Tennis
- https://ec.europa.eu/eurostat/web/rss
- https://github.com/logpai/loghub
- https://github.com/jdorfman/awesome-json-datasets?tab=readme-ov-file
- https://koxudaxi.github.io/datamodel-code-generator/
- https://github.com/2U1/Qwen2-VL-Finetune/blob/master/scripts/finetune.sh