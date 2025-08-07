package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"boxer/internal/predictor"

	"golang.org/x/net/websocket"
)

func main() {
	predictorAddr := "localhost:50051"
	serverAddr := ":8080"
	client := predictor.NewPredictorClient(predictorAddr)
	handler := NewHandler(client)

	http.Handle("/ws", websocket.Handler(handler.wsHandler))
	http.Handle("/", http.FileServer(http.Dir("./static")))

	fmt.Println("Server running at", serverAddr)
	log.Fatal(http.ListenAndServe(serverAddr, nil))
}

type Predictor interface {
	Predict(ctx context.Context, payload []byte, source string) (predictor.Boxes, error)
}

func NewHandler(predictor Predictor) Handler {
	return Handler{predictor: predictor}
}

type Handler struct {
	predictor Predictor
}

type Message struct {
	Image     []byte
	Timestamp time.Time
}

func (h *Handler) wsHandler(ws *websocket.Conn) {
	defer ws.Close()

	var (
		latestMu sync.Mutex
		latest   Message
	)

	// Read loop (per connection)
	go func() {
		for {
			var img []byte
			err := websocket.Message.Receive(ws, &img)
			if err != nil {
				log.Println("WebSocket read error:", err)
				break
			}

			msg := Message{Image: img, Timestamp: time.Now()}
			latestMu.Lock()
			latest = msg
			latestMu.Unlock()
		}
	}()

	// Process loop (per connection)
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()

	for range ticker.C {
		latestMu.Lock()
		msg := latest
		latestMu.Unlock()

		if msg.Image == nil {
			continue
		}

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		boxes, err := h.predictor.Predict(ctx, msg.Image, "webcam")
		log.Printf("Time to receive prediction %v", time.Since(msg.Timestamp))
		cancel()

		if err != nil {
			log.Println("gRPC error:", err)
			continue
		}

		log.Println("objects found", boxes.GetLabels())
		data, err := json.Marshal(boxes)
		if err != nil {
			log.Println("JSON marshal error:", err)
			continue
		}
		if err := websocket.Message.Send(ws, string(data)); err != nil {
			log.Println("WebSocket send error:", err)
			return
		}
		log.Printf("Time for iteration %v", time.Since(msg.Timestamp))
	}
}
