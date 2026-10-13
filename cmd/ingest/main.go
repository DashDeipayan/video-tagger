// Command ingest is the write-side entry point of the pipeline.
//
// Responsibilities, and deliberately nothing more:
//   - accept a video (uploaded bytes, or a reference to a file already on disk)
//   - persist the blob and a metadata row
//   - publish video.ingested and return immediately
//
// It never decodes and never runs a model. A request returns in milliseconds
// regardless of how long the video is, because the expensive work happens
// downstream. That asynchronous boundary is the whole reason the system can
// absorb bursts: upload capacity and processing capacity scale separately.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/instawork/video-tagging/internal/bus"
	"github.com/instawork/video-tagging/internal/config"
	"github.com/instawork/video-tagging/internal/frames"
	"github.com/instawork/video-tagging/internal/httpx"
	"github.com/instawork/video-tagging/internal/model"
	"github.com/instawork/video-tagging/internal/store"
)

type server struct {
	cfg config.Config
	st  store.Store
	b   bus.Bus
	log *slog.Logger
}

func main() {
	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	cfg := config.Load()

	if err := os.MkdirAll(cfg.BlobDir, 0o755); err != nil {
		log.Error("blob dir", "err", err)
		os.Exit(1)
	}
	st, err := store.NewSQLite(cfg.DBPath)
	if err != nil {
		log.Error("store", "err", err)
		os.Exit(1)
	}
	defer st.Close()

	b := newBus(cfg, log)
	defer b.Close()

	s := &server{cfg: cfg, st: st, b: b, log: log}

	// Go 1.22+ pattern routing covers everything here; a third-party router
	// would be dependency surface for no benefit.
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/videos", s.handleUpload)
	mux.HandleFunc("POST /api/videos/register", s.handleRegister)
	mux.HandleFunc("GET /api/videos", s.handleList)
	mux.HandleFunc("GET /api/videos/{id}", s.handleGet)
	mux.HandleFunc("GET /api/stats", s.handleStats)
	mux.HandleFunc("GET /api/health", s.handleHealth)
	mux.HandleFunc("GET /media/{id}", s.handleMedia)
	mux.Handle("GET /frames/", http.StripPrefix("/frames/", http.FileServer(http.Dir(cfg.FrameDir))))
	mux.Handle("/", http.FileServer(http.Dir(cfg.WebDir)))

	srv := &http.Server{
		Addr:    cfg.IngestAddr,
		Handler: httpx.CORS(httpx.Log(log, mux)),
	}

	go func() {
		log.Info("ingest listening", "addr", cfg.IngestAddr, "ui", "http://localhost"+cfg.IngestAddr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("listen", "err", err)
			os.Exit(1)
		}
	}()

	waitForShutdown(log)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	srv.Shutdown(ctx)
}

// handleUpload accepts a multipart file upload.
//
// The bytes are streamed straight to the blob store rather than buffered in
// memory, so a 5 GB video costs the same resident memory as a 5 MB one.
func (s *server) handleUpload(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		httpx.Error(w, http.StatusBadRequest, "expected multipart form with a 'file' field")
		return
	}
	file, hdr, err := r.FormFile("file")
	if err != nil {
		httpx.Error(w, http.StatusBadRequest, "missing 'file' field")
		return
	}
	defer file.Close()

	id := uuid.NewString()
	dst := filepath.Join(s.cfg.BlobDir, id+filepath.Ext(hdr.Filename))
	out, err := os.Create(dst)
	if err != nil {
		httpx.Error(w, http.StatusInternalServerError, err.Error())
		return
	}
	written, err := io.Copy(out, file)
	out.Close()
	if err != nil {
		httpx.Error(w, http.StatusInternalServerError, err.Error())
		return
	}
	s.accept(w, r, id, hdr.Filename, dst, written)
}

// handleRegister ingests a file already on disk by reference.
//
// At a million hours a month almost nothing is uploaded through the API: video
// lands in object storage from edge devices and the pipeline is told where it
// is. This endpoint models that path, and is what the demo uses for the sample
// corpus.
func (s *server) handleRegister(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Path string `json:"path"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Path == "" {
		httpx.Error(w, http.StatusBadRequest, "expected {\"path\": \"...\"}")
		return
	}
	abs, err := filepath.Abs(req.Path)
	if err != nil {
		httpx.Error(w, http.StatusBadRequest, err.Error())
		return
	}
	fi, err := os.Stat(abs)
	if err != nil {
		httpx.Error(w, http.StatusBadRequest, "not readable: "+abs)
		return
	}
	s.accept(w, r, uuid.NewString(), filepath.Base(abs), abs, fi.Size())
}

// accept is the shared tail of both ingest paths: probe, persist, publish.
func (s *server) accept(w http.ResponseWriter, r *http.Request, id, filename, path string, size int64) {
	ctx := r.Context()

	// ffprobe reads container headers only — no pictures are decoded — so this
	// is cheap enough to do synchronously and gives the UI a duration to show
	// immediately.
	p, err := frames.ProbeVideo(ctx, path)
	if err != nil {
		s.log.Warn("probe failed, continuing", "path", path, "err", err)
	}

	v := &model.Video{
		ID: id, Filename: filename, Path: path, SizeBytes: size,
		DurationSec: p.DurationSec, Width: p.Width, Height: p.Height,
		Status: model.StatusPending,
	}
	if err := s.st.CreateVideo(ctx, v); err != nil {
		httpx.Error(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Keyed by video ID so every event for one video lands on one partition and
	// is therefore processed in order.
	if err := s.b.Publish(ctx, bus.TopicIngested, id, model.VideoIngested{
		VideoID: id, Path: path, Filename: filename,
		SizeBytes: size, DurationSec: p.DurationSec,
	}); err != nil {
		s.st.SetStatus(ctx, id, model.StatusFailed, "enqueue: "+err.Error())
		httpx.Error(w, http.StatusInternalServerError, err.Error())
		return
	}

	s.log.Info("ingested", "id", id, "file", filename, "dur", fmt.Sprintf("%.1fs", p.DurationSec))
	// 202: accepted for processing, not yet processed. The client polls
	// GET /api/videos/{id} to watch it move through the pipeline.
	httpx.JSON(w, http.StatusAccepted, v)
}

func (s *server) handleGet(w http.ResponseWriter, r *http.Request) {
	v, err := s.st.GetVideo(r.Context(), r.PathValue("id"))
	if err != nil {
		httpx.Error(w, http.StatusNotFound, "no such video")
		return
	}
	httpx.JSON(w, http.StatusOK, v)
}

func (s *server) handleList(w http.ResponseWriter, r *http.Request) {
	limit := 100
	if l, err := strconv.Atoi(r.URL.Query().Get("limit")); err == nil && l > 0 {
		limit = l
	}
	vs, err := s.st.ListVideos(r.Context(), limit)
	if err != nil {
		httpx.Error(w, http.StatusInternalServerError, err.Error())
		return
	}
	if vs == nil {
		vs = []*model.Video{}
	}
	httpx.JSON(w, http.StatusOK, vs)
}

func (s *server) handleStats(w http.ResponseWriter, r *http.Request) {
	st, err := s.st.Stats(r.Context())
	if err != nil {
		httpx.Error(w, http.StatusInternalServerError, err.Error())
		return
	}
	httpx.JSON(w, http.StatusOK, st)
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	httpx.JSON(w, http.StatusOK, map[string]any{"service": "ingest", "status": "ok"})
}

// handleMedia streams a video back for playback in the UI. http.ServeFile
// handles Range requests, so the browser can seek without downloading the file.
func (s *server) handleMedia(w http.ResponseWriter, r *http.Request) {
	v, err := s.st.GetVideo(r.Context(), r.PathValue("id"))
	if err != nil {
		httpx.Error(w, http.StatusNotFound, "no such video")
		return
	}
	http.ServeFile(w, r, v.Path)
}

// newBus selects the transport. Kafka when brokers are configured, otherwise
// the in-process implementation so the system still runs with nothing else up.
// newBus requires Kafka. There is deliberately no in-process fallback: the
// stages are separate processes, so a bus local to ingest would have no
// subscribers, and we would accept uploads with 202 and silently drop them.
// Failing to start is the correct behaviour — losing video is not.
func newBus(cfg config.Config, log *slog.Logger) bus.Bus {
	if len(cfg.KafkaBrokers) == 0 {
		log.Error("no kafka brokers configured; ingest cannot accept video it cannot enqueue")
		os.Exit(1)
	}
	log.Info("using kafka", "brokers", strings.Join(cfg.KafkaBrokers, ","))
	return bus.NewKafka(cfg.KafkaBrokers, log)
}

func waitForShutdown(log *slog.Logger) {
	c := make(chan os.Signal, 1)
	signal.Notify(c, syscall.SIGINT, syscall.SIGTERM)
	<-c
	log.Info("shutting down")
}
