// Package frames samples still images out of a video with ffmpeg.
//
// This stage is where the 1,000,000 hours/month requirement is actually won or
// lost. Running a model on every frame would mean ~41,000 inferences/second at
// that volume, which is not buildable. Sampling makes it ~700/second, which is
// roughly one GPU. Scene and environment content barely changes between
// adjacent frames, so 29 out of every 30 carry almost no new information.
package frames

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/instawork/video-tagging/internal/model"
)

// Probe is the technical metadata ffprobe reports about a file.
type Probe struct {
	DurationSec float64
	Width       int
	Height      int
	NBFrames    int
}

// ProbeVideo reads container metadata without decoding any pictures.
func ProbeVideo(ctx context.Context, path string) (Probe, error) {
	out, err := exec.CommandContext(ctx, "ffprobe",
		"-v", "error",
		"-select_streams", "v:0",
		"-show_entries", "format=duration:stream=width,height,nb_frames",
		"-of", "json", path).Output()
	if err != nil {
		return Probe{}, fmt.Errorf("ffprobe %s: %w", filepath.Base(path), err)
	}
	var raw struct {
		Format  struct{ Duration string } `json:"format"`
		Streams []struct {
			Width, Height int
			NBFrames      string `json:"nb_frames"`
		} `json:"streams"`
	}
	if err := json.Unmarshal(out, &raw); err != nil {
		return Probe{}, err
	}
	var p Probe
	p.DurationSec, _ = strconv.ParseFloat(raw.Format.Duration, 64)
	if len(raw.Streams) > 0 {
		p.Width, p.Height = raw.Streams[0].Width, raw.Streams[0].Height
		p.NBFrames, _ = strconv.Atoi(raw.Streams[0].NBFrames)
	}
	return p, nil
}

// Sampler extracts frames according to a sampling policy.
type Sampler struct {
	// EverySec is the target spacing between sampled frames.
	EverySec float64
	// MaxFrames caps work for very long videos, so a 4-hour file cannot occupy a
	// worker indefinitely and stall its partition.
	MaxFrames int
	// KeyframesOnly decodes only I-frames instead of the full stream. On this
	// corpus keyframes are ~0.4% of all frames, so this is roughly a 250x
	// reduction in decode cost — usually the difference between being CPU-bound
	// and being GPU-bound.
	KeyframesOnly bool
	// OutDir is where extracted JPEGs are written.
	OutDir string
}

func DefaultSampler(outDir string) *Sampler {
	return &Sampler{EverySec: 2.0, MaxFrames: 24, KeyframesOnly: false, OutDir: outDir}
}

// Extract samples videoPath and returns the frames it wrote to disk.
//
// Frames are written as small JPEGs (224px short edge) because that is the
// resolution CLIP consumes anyway — shrinking here rather than in the model
// saves both disk and transfer to the inference service.
func (s *Sampler) Extract(ctx context.Context, videoID, videoPath string, p Probe) ([]model.Frame, error) {
	dir := filepath.Join(s.OutDir, videoID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}

	// Choose an fps that yields at most MaxFrames over the whole video, so long
	// videos get sampled more sparsely rather than producing unbounded work.
	fps := 1.0 / s.EverySec
	if p.DurationSec > 0 {
		if want := float64(s.MaxFrames) / p.DurationSec; want < fps {
			fps = want
		}
	}

	filter := fmt.Sprintf("fps=%f,scale=224:224:force_original_aspect_ratio=increase,crop=224:224", fps)
	args := []string{"-v", "error"}
	if s.KeyframesOnly {
		// Must precede -i: tells the decoder to discard non-key frames outright.
		args = append(args, "-skip_frame", "nokey")
	}
	args = append(args,
		"-i", videoPath,
		"-vf", filter,
		"-fps_mode", "vfr", // -vsync was removed in ffmpeg 8; this is the replacement
		"-frame_pts", "1",
		"-q:v", "4",
		filepath.Join(dir, "frame_%04d.jpg"),
	)

	cmd := exec.CommandContext(ctx, "ffmpeg", args...)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg: %w: %s", err, stderr.String())
	}

	entries, err := filepath.Glob(filepath.Join(dir, "frame_*.jpg"))
	if err != nil {
		return nil, err
	}
	if len(entries) == 0 {
		return nil, fmt.Errorf("no frames extracted from %s", filepath.Base(videoPath))
	}

	// Reconstruct each frame's timestamp from its ordinal and the sampling rate.
	out := make([]model.Frame, 0, len(entries))
	for i, f := range entries {
		out = append(out, model.Frame{
			VideoID: videoID,
			Index:   i,
			TSSec:   float64(i) / fps,
			Path:    f,
		})
	}
	return out, nil
}
