import os
from datetime import datetime

import phoenix as px

PROJECT = os.getenv("PHOENIX_PROJECT_NAME", "travel-agent")

client = px.Client(endpoint="http://localhost:6006")
spans = client.get_spans_dataframe(project_name=PROJECT)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output = f"spanexport_{PROJECT}_{timestamp}.csv"
spans.to_csv(output, index=False)
print(f"Exported {len(spans)} spans to {output}")
