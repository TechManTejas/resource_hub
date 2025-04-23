from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

# Connect to Milvus
connections.connect(host='localhost', port='19530')

# Check connection and server version
print(f"Server version: {utility.get_server_version()}")

# Create a simple collection for testing
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields, "Simple test collection")

# Drop the collection if it already exists
if utility.has_collection("test_collection"):
    utility.drop_collection("test_collection")
    
collection = Collection("test_collection", schema)
print("Collection created successfully!")

# Insert some data
import random
entities = [
    [i for i in range(10)],  # ids
    [[random.random() for _ in range(128)] for _ in range(10)]  # vectors
]

insert_result = collection.insert(entities)
print(f"Inserted {insert_result.insert_count} entities")

# Create an index
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index("vector", index_params)
print("Index created")

# Load the collection for search
collection.load()

# Search for a vector
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(
    [entities[1][0]],  # Use the first vector as query
    "vector",
    search_params,
    limit=3
)
print("Search complete")
print(f"First result ID: {results[0][0].id}, distance: {results[0][0].distance}")

# Clean up
connections.disconnect("default")
print("Test completed successfully!")