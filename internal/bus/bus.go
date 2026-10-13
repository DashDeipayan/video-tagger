// Package bus is the messaging seam between services.
//
// Every stage of the pipeline talks to its neighbours through this interface
// and never through a direct call. That is what lets each stage scale on its
// own hardware: the decoder is CPU-bound and the tagger is GPU-bound, so in
// production they are different autoscaling groups with different instance
// types. Swapping the implementation changes the transport, not the code.
package bus

import (
	"context"
	"encoding/json"
)

// Topics carried by the pipeline. Each is produced by exactly one service and
// consumed by exactly one consumer group, which keeps ownership unambiguous.
const (
	TopicIngested = "video.ingested"         // ingest  -> decoder
	TopicFrames   = "video.frames.extracted" // decoder -> tagger
	TopicTagged   = "video.tagged"           // tagger  -> search (index projection)
)

// Handler processes one message. Returning an error means the message is not
// committed, so it will be redelivered — handlers must therefore be idempotent.
type Handler func(ctx context.Context, key string, payload []byte) error

// Bus is the transport abstraction. Kafka is the only implementation: the
// stages are separate deployables, so the transport has to cross process
// boundaries. The interface exists so that swapping Kafka for SQS, Pub/Sub or
// NATS is a one-file change rather than a rewrite of every consumer.
type Bus interface {
	Publish(ctx context.Context, topic, key string, v any) error
	Subscribe(ctx context.Context, topic, group string, h Handler) error
	Close() error
}

// PublishJSON marshals v and publishes it. Shared by every implementation so
// the wire format is identical regardless of transport.
func marshal(v any) ([]byte, error) {
	if b, ok := v.([]byte); ok {
		return b, nil
	}
	return json.Marshal(v)
}
