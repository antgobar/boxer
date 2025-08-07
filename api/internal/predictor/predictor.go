package predictor

import (
	"context"
	"log"

	pb "boxer/internal/genproto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type Predictor struct {
	client     pb.ModelClient
	connection *grpc.ClientConn
}

type Box struct {
	X1    float32 `json:"x1"`
	Y1    float32 `json:"y1"`
	X2    float32 `json:"x2"`
	Y2    float32 `json:"y2"`
	Score float32 `json:"score"`
	Label string  `json:"label"`
}

type Boxes []Box

func (boxes Boxes) GetLabels() []string {
	var labels []string
	for _, box := range boxes {
		labels = append(labels, box.Label)
	}
	return labels
}

func (p *Predictor) Close() {
	p.connection.Close()
}

func (p *Predictor) Predict(ctx context.Context, payload []byte, source string) (Boxes, error) {
	response, err := p.client.Predict(ctx, &pb.PredictRequest{
		Payload: payload,
		Source:  source,
	})

	if err != nil {
		return nil, err
	}

	var boxes []Box
	for _, item := range response.Items {
		boxes = append(boxes, Box{
			X1:    item.X1,
			Y1:    item.Y1,
			X2:    item.X2,
			Y2:    item.Y2,
			Score: item.Score,
			Label: item.Label,
		})
	}
	return boxes, nil
}

func NewPredictorClient(addr string) *Predictor {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}

	client := pb.NewModelClient(conn)

	return &Predictor{client, conn}
}
