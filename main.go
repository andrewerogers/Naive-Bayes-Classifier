package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
)

const(
	decisionThreshold = 0.5
)

func main() {
	nb := newClassifier()
	dataset := dataset("./sentiment labelled sentences/amazon_cells_labelled.txt")
	nb.train(dataset)

	// Prompt for inputs from console
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("Enter your review: ")
		sentence, _ := reader.ReadString('\n')

		pos, neg := nb.classify(sentence)
		diff := math.Abs(pos - neg)
		class := ""
		if pos >= neg || diff < decisionThreshold {
			class = positive
		} else {
			class = negative
		}
		fmt.Printf("> Your review is %s\n", class)
		fmt.Printf("> Probability distribution - positive: %f negative: %f \n", pos, neg)
	}
}

// dataset returns a map of sentences to their classes from a file
func dataset(file string) map[string]string {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	dataset := make(map[string]string)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		l := scanner.Text()
		data := strings.Split(l, "\t")
		if len(data) != 2 {
			continue
		}
		sentence := data[0]
		if data[1] == "0" {
			dataset[sentence] = negative
		} else if data[1] == "1" {
			dataset[sentence] = positive
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
	return dataset
}