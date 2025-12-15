package server

import (
	"context"
	"io"
	"log"
	"time"

	pb "go_backend/proto"
	"go_backend/internal/client"
)

type StreamServer struct {
	pb.UnimplementedStreamASRServer
}

func (s *StreamServer) StreamAudio(stream pb.StreamASR_StreamAudioServer) error {
	ctx := context.Background()
	ws, err := client.NewWSClient(ctx, "ws://localhost:8787/ws/stream")
	if err != nil {
		return err
	}
	go ws.Listen(ctx, stream, "session-1")

	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			log.Println("Recv error:", err)
			return err
		}

		sendCtx, cancel := context.WithTimeout(ctx, 1*time.Second)
		_ = ws.SendAudio(sendCtx, chunk.AudioData)
		cancel()
	}
}