import phoenix as px

client = px.Client(endpoint="http://localhost:6006")
spans = client.get_spans_dataframe(project_name="travel-agent")

output = "spans.csv"
spans.to_csv(output, index=False)
print(f"Exported {len(spans)} spans to {output}")
