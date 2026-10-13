package bus

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"sync"
	"time"

	"github.com/segmentio/kafka-go"
)

// KafkaBus is the production transport.
//
// Partitioning is keyed by video ID, which gives us per-video ordering (a
// video's frames can never be tagged before they are extracted) while still
// allowing the whole topic to be consumed in parallel. Adding partitions and
// consumer instances is the horizontal scaling lever for every stage.
type KafkaBus struct {
	brokers []string
	mu      sync.Mutex
	writers map[string]*kafka.Writer
	readers []*kafka.Reader
	log     *slog.Logger
}

func NewKafka(brokers []string, log *slog.Logger) *KafkaBus {
	return &KafkaBus{
		brokers: brokers,
		writers: make(map[string]*kafka.Writer),
		log:     log,
	}
}

func (k *KafkaBus) writer(topic string) *kafka.Writer {
	k.mu.Lock()
	defer k.mu.Unlock()
	if w, ok := k.writers[topic]; ok {
		return w
	}
	w := &kafka.Writer{
		Addr:                   kafka.TCP(k.brokers...),
		Topic:                  topic,
		Balancer:               &kafka.Hash{}, // key -> partition, for per-video ordering
		AllowAutoTopicCreation: true,
		BatchTimeout:           10 * time.Millisecond, // keep the demo feeling live
		RequiredAcks:           kafka.RequireOne,
	}
	k.writers[topic] = w
	return w
}

func (k *KafkaBus) Publish(ctx context.Context, topic, key string, v any) error {
	b, err := marshal(v)
	if err != nil {
		return err
	}
	return k.writer(topic).WriteMessages(ctx, kafka.Message{
		Key:   []byte(key),
		Value: b,
	})
}

// Subscribe runs a consumer loop until ctx is cancelled.
//
// Offsets are committed only after the handler succeeds, giving at-least-once
// delivery. Combined with idempotent handlers (every write is an upsert keyed
// by video ID) that yields effectively-once processing without the cost of a
// transactional protocol.
func (k *KafkaBus) Subscribe(ctx context.Context, topic, group string, h Handler) error {
	r := kafka.NewReader(kafka.ReaderConfig{
		Brokers:     k.brokers,
		Topic:       topic,
		GroupID:     group,
		MinBytes:    1,
		MaxBytes:    10e6,
		MaxWait:     200 * time.Millisecond,
		StartOffset: kafka.FirstOffset,
	})
	k.mu.Lock()
	k.readers = append(k.readers, r)
	k.mu.Unlock()

	go func() {
		defer r.Close()
		for {
			m, err := r.FetchMessage(ctx)
			if err != nil {
				if ctx.Err() != nil || errors.Is(err, io.EOF) {
					return
				}
				k.log.Error("fetch failed", "topic", topic, "err", err)
				time.Sleep(time.Second)
				continue
			}
			if err := h(ctx, string(m.Key), m.Value); err != nil {
				// Not committed: Kafka will redeliver. A production build adds a
				// retry counter here and routes to a dead-letter topic past N
				// attempts, so one poison video cannot stall a partition.
				k.log.Error("handler failed, will redeliver", "topic", topic, "key", string(m.Key), "err", err)
				continue
			}
			if err := r.CommitMessages(ctx, m); err != nil && ctx.Err() == nil {
				k.log.Error("commit failed", "topic", topic, "err", err)
			}
		}
	}()
	return nil
}

func (k *KafkaBus) Close() error {
	k.mu.Lock()
	defer k.mu.Unlock()
	for _, w := range k.writers {
		w.Close()
	}
	for _, r := range k.readers {
		r.Close()
	}
	return nil
}
