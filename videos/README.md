# Sample corpus

The videos used to develop and benchmark this system are **not committed** —
they are real first-person footage of identifiable people.

To run the demo, drop any `.mp4` files in this directory and run:

```bash
make up
make demo      # registers every videos/*.mp4, waits for tagging, runs searches
```

Or upload through the UI at http://localhost:8080 (drag and drop).

## What the system was tuned against

Four egocentric ("body-cam" perspective) clips of manual work, 9.1 minutes total,
360p H.264 with AAC audio:

| # | Content | Environment | Activity |
|---|---|---|---|
| 1 | making a bed — sheets, pillows | bedroom | bed-making |
| 2 | sorting small items on a steel line | factory floor | sorting |
| 3 | washing stove grates in a sink | kitchen | dishwashing |
| 4 | applying cement overhead with a trowel | construction site | plastering |

The default taxonomy in `internal/taxonomy/seed.go` and the first-person prompt
phrasings in it are tuned for this kind of workforce footage. For a different
corpus, that file is the thing to edit — or add terms at runtime through the
taxonomy service, which needs no redeploy.
