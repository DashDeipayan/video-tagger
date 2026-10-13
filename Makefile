# Video Tagging & Search — developer entry points.
#
# Services are separate binaries that communicate only over Kafka. They can be
# started, stopped and scaled independently; `make up` just starts one of each
# for convenience.

SHELL     := /bin/bash
KAFKA     := /opt/homebrew/opt/kafka/bin
PY        := sidecar/.venv/bin
RUN       := .run
export PATH := $(KAFKA):$(PATH)

.PHONY: help build up down logs kafka sidecar topics seed demo clean status taxonomy

help:
	@echo "make kafka     — start the local Kafka broker (KRaft, no ZooKeeper)"
	@echo "make taxonomy  — show the taxonomy DAG"
	@echo "make sidecar   — start the CLIP inference sidecar"
	@echo "make up        — build and start all four Go services"
	@echo "make seed      — register the sample videos in ./videos"
	@echo "make demo      — seed, wait for tagging, then run example searches"
	@echo "make fresh     — restart with an empty catalog"
	@echo "make status    — show what is running"
	@echo "make down      — stop the Go services and the sidecar"
	@echo "make clean     — stop everything and delete the catalog + frames"

build:
	@go build -o bin/taxonomy ./cmd/taxonomy
	@go build -o bin/ingest   ./cmd/ingest
	@go build -o bin/decoder  ./cmd/decoder
	@go build -o bin/tagger   ./cmd/tagger
	@go build -o bin/search   ./cmd/search
	@echo "✓ built bin/{taxonomy,ingest,decoder,tagger,search}"

kafka:
	@mkdir -p $(RUN) data
	@# Format the KRaft metadata dir on first run. --ignore-formatted makes this
	@# idempotent, so the same target works cold and warm.
	@if [ ! -d data/kafka-logs ]; then \
		$(KAFKA)/kafka-storage format -t $$($(KAFKA)/kafka-storage random-uuid) \
			-c deploy/kafka-kraft.properties --ignore-formatted >/dev/null 2>&1; \
		echo "  formatted kafka storage"; \
	fi
	@if nc -z localhost 9092 2>/dev/null; then echo "✓ kafka already up"; else \
		(nohup $(KAFKA)/kafka-server-start deploy/kafka-kraft.properties > $(RUN)/kafka.log 2>&1 < /dev/null &); \
		echo "starting kafka..."; \
		for i in $$(seq 1 30); do nc -z localhost 9092 2>/dev/null && break; sleep 1; done; \
		echo "✓ kafka up on :9092"; fi

topics:
	@for t in video.ingested video.frames.extracted video.tagged; do \
		$(KAFKA)/kafka-topics --bootstrap-server localhost:9092 \
			--create --if-not-exists --topic $$t --partitions 3 --replication-factor 1 >/dev/null 2>&1; \
	done
	@$(KAFKA)/kafka-topics --bootstrap-server localhost:9092 --list | grep video

sidecar:
	@mkdir -p $(RUN)
	@if curl -sf -m 2 localhost:8501/health >/dev/null 2>&1; then echo "✓ sidecar already up"; else \
		(cd sidecar && nohup .venv/bin/uvicorn main:app --host 127.0.0.1 --port 8501 > ../$(RUN)/sidecar.log 2>&1 < /dev/null &) ; \
		echo "starting CLIP sidecar (loads model weights)..."; \
		for i in $$(seq 1 60); do curl -sf -m 2 localhost:8501/health >/dev/null 2>&1 && break; sleep 1; done; \
		curl -s localhost:8501/health; echo; fi

up: build kafka topics sidecar
	@mkdir -p $(RUN) data
	@# Each service is fully detached (stdin/stdout/stderr redirected) so make
	@# returns immediately instead of waiting on the children's inherited pipe.
	@# taxonomy first: the tagger and search both block on its snapshot at boot.
	@(nohup ./bin/taxonomy > $(RUN)/taxonomy.log 2>&1 < /dev/null & echo $$! > $(RUN)/taxonomy.pid)
	@for i in $$(seq 1 30); do curl -sf -m 1 localhost:8082/api/health >/dev/null 2>&1 && break; sleep 1; done
	@for s in decoder tagger search ingest; do \
		(nohup ./bin/$$s > $(RUN)/$$s.log 2>&1 < /dev/null & echo $$! > $(RUN)/$$s.pid); \
	done
	@sleep 3
	@echo ""
	@echo "  ✓ pipeline up"
	@echo "    UI        http://localhost:8080"
	@echo "    ingest    http://localhost:8080/api"
	@echo "    search    http://localhost:8081/api/search"
	@echo "    taxonomy  http://localhost:8082/api/taxonomy/tree"
	@echo ""

seed:
	@for f in videos/*.mp4; do \
		curl -s -X POST localhost:8080/api/videos/register \
			-H 'Content-Type: application/json' \
			-d "{\"path\":\"$$PWD/$$f\"}" | python3 -c 'import sys,json; d=json.load(sys.stdin); print("  registered", d.get("filename"), d.get("id","")[:8])'; \
	done

demo:
	@python3 scripts/demo.py

# Re-run the demo against an empty catalog. Registering a path again creates a
# new video by design (re-ingest is a real operation), so repeated demo runs
# accumulate rows unless the catalog is reset.
fresh:
	@$(MAKE) --no-print-directory down >/dev/null 2>&1 || true
	@rm -f data/catalog.db* && rm -rf data/frames
	@$(MAKE) --no-print-directory up

status:
	@printf "kafka    "; nc -z localhost 9092 2>/dev/null && echo "✓ :9092" || echo "✗ down"
	@printf "sidecar  "; curl -sf -m 2 localhost:8501/health 2>/dev/null | head -c 80 || echo "✗ down"; echo
	@printf "ingest   "; curl -sf -m 2 localhost:8080/api/health >/dev/null 2>&1 && echo "✓ :8080" || echo "✗ down"
	@printf "search   "; curl -sf -m 2 localhost:8081/api/health >/dev/null 2>&1 && echo "✓ :8081" || echo "✗ down"
	@printf "taxonomy "; curl -sf -m 2 localhost:8082/api/health 2>/dev/null | head -c 80 || echo "✗ down"; echo

down:
	@-for p in $(RUN)/*.pid; do [ -f "$$p" ] && kill $$(cat $$p) 2>/dev/null; rm -f $$p; done
	@# Also sweep by name: a stale or missing pidfile would otherwise leave a
	@# service running, and a second copy competing for the same port and
	@# consumer group is a confusing way to lose an afternoon.
	@-for s in ingest decoder tagger search taxonomy; do pkill -f "bin/$$s" 2>/dev/null; done; true
	@-pkill -f 'uvicorn main:app' 2>/dev/null || true
	@echo "✓ services stopped (kafka left running; use 'make clean' to stop it)"

clean: down
	@-pkill -f kafka-server-start 2>/dev/null || true
	@sleep 2
	@rm -rf data bin $(RUN)
	@echo "✓ cleaned (catalog, blobs, frames, kafka log dir and binaries removed)"

taxonomy:
	@curl -s localhost:8082/api/taxonomy/tree | python3 scripts/tree.py
