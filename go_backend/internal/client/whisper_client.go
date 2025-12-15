package client

import (
	"context"
	"encoding/json"
	"log"
	"time"

	pb "go_backend/proto"
	"nhooyr.io/websocket"
)

type WSClient struct {
	conn *websocket.Conn
}

// Connect to Python WebSocket server
func NewWSClient(ctx context.Context, url string) (*WSClient, error) {
	c, _, err := websocket.Dial(ctx, url, nil)
	if err != nil {
		return nil, err
	}
	log.Println("🔌 Connected to Python ASR websocket")
	return &WSClient{conn: c}, nil
}

// Send PCM bytes to Python ASR
func (w *WSClient) SendAudio(ctx context.Context, data []byte) error {
	return w.conn.Write(ctx, websocket.MessageBinary, data)
}

// Listen for ASR messages and forward them as gRPC transcripts
func (w *WSClient) Listen(ctx context.Context, stream pb.StreamASR_StreamAudioServer, sessionID string) {
	for {
		_, msg, err := w.conn.Read(ctx)
		if err != nil {
			log.Println("ASR stream closed:", err)
			return
		}

		var packet map[string]interface{}
		if err := json.Unmarshal(msg, &packet); err != nil {
			continue
		}

		text, _ := packet["text"].(string)
		if text == "" {
			continue
		}

		resp := &pb.Transcript{
			Text:      text,
			SessionId: sessionID,
			Timestamp: float32(time.Now().UnixMilli()) / 1000,
		}

		if err := stream.Send(resp); err != nil {
			log.Println("Send error:", err)
			return
		}
	}
}