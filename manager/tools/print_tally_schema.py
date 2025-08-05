def print_tally_schema(schema):
    """
    Prints the schema of the TallyDB database in a readable format.

    Parameters:
    schema (dict): A dictionary representing the database schema.
    """
    for table, details in schema.items():
        print(f"Table: {table}")
        print("Columns:")
        for column in details['columns']:
            print(f" - {column['name']} ({column['type']})")
        print("Relationships:")
        for relationship in details.get('relationships', []):
            print(f" - {relationship['table']} (type: {relationship['type']})")
        print("\n")  # Add a newline for better readability

# Example usage:
# schema = {
#     'users': {
#         'columns': [{'name': 'id', 'type': 'INTEGER'}, {'name': 'name', 'type': 'TEXT'}],
#         'relationships': [{'table': 'orders', 'type': 'one-to-many'}]
#     },
#     'orders': {
#         'columns': [{'name': 'id', 'type': 'INTEGER'}, {'name': 'user_id', 'type': 'INTEGER'}],
#         'relationships': []
#     }
# }
# print_tally_schema(schema)