# ECHO UML diagrams

These diagrams use [PlantUML](https://plantuml.com/). They document the current
asynchronous runtime rather than the deprecated synchronous inference routes.

| Diagram | Purpose |
| --- | --- |
| `system-deployment.puml` | Runtime containers, queues, persistence, and optional accelerator workers. |
| `application-components.puml` | Frontend and backend responsibilities and their dependencies. |
| `async-job-sequence.puml` | Upload, dispatch, execution, polling, cancellation, and cleanup flow. |

Render all diagrams from this directory with:

```bash
plantuml -tsvg *.puml
```

PlantUML files are intentionally the source of truth; generated SVG or PNG
files are not committed.
