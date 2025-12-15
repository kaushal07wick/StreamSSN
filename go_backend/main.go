package main

import (
	"log"
	"net"

	pb "go_backend/proto"
	"go_backend/internal/server"
	"google.golang.org/grpc"
)

func main() {
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	s := grpc.NewServer()
	pb.RegisterStreamASRServer(s, &server.StreamServer{})

	log.Println("✅ StreamSSN gRPC backend running on :50051")
	if err := s.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}
