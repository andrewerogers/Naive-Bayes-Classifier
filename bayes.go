package main

import "math"

const (
	positive = "positive"
	negative = "negative"
	alpha = 1
)

// wordFrequency stores frequency of words by class
type wordFrequency struct {
	word    string
	counter map[string]int
}

// classifier can be trained and used to categorize object
// dataset contains sentence organized into each class
// words contains a map of word with positive / negative frequency for each word
type classifier struct {
	dataset map[string][]string
	words   map[string]wordFrequency
}

// newClassifier returns a new classifier with empty dataset and words
func newClassifier() *classifier {
	c := new(classifier)
	c.dataset = map[string][]string{
		positive: []string{},
		negative: []string{},
	}
	c.words = map[string]wordFrequency{}
	return c
}

// train populates a classifier's dataset and words with input dataset map
// Sample dataset: map[string]string{
//	"The restaurant is excellent": "Positive",
//	"I really love this restaurant": "Positive",
//	"Their food is awful": "Negative",
//}
func (c *classifier) train(dataset map[string]string) {
	for sentence, class := range dataset {
		c.dataset[class] = append(c.dataset[class], sentence)
		words := tokenize(sentence)
		for _, w := range words {
			wf, ok := c.words[w]
			if !ok {
				wf = wordFrequency{word: w, counter: map[string]int{
					positive: 0,
					negative: 0,
				}}
			}
			wf.counter[class]++
			c.words[w] = wf
		}
	}
}

// classify returns probability of a sentence being part of a given
// class. i.e. P(positive) = 0.2, P(negative) = 0.5
func (c classifier) classify(sentence string) (float64, float64) {
	words := tokenize(sentence)
	posProb := c.prob(words, positive)
	negProb := c.prob(words, negative)
	return posProb, negProb
}

// prior returns the prior probability of each class of the classifier
// This probability is determined purely by the training dataset
func (c classifier) prior(class string) float64 {
	return float64(len(c.dataset[class])) / float64(len(c.dataset[positive])+len(c.dataset[negative]))
}

// totalWordCount returns the word count of a class (duplicated also count)
// If class provided is not positive or negative, it returns
// the total word count in dataset.
func (c classifier) totalWordCount(class string) int {
	posCount := 0
	negCount := 0
	for _, wf := range c.words {
		posCount += wf.counter[positive]
		negCount += wf.counter[negative]
	}
	if class == positive {
		return posCount
	} else if class == negative {
		return negCount
	} else {
		return posCount + negCount
	}
}

// prob retuns the laplace-smoothed probability of a list of words being in a class
func (c classifier) prob(words []string, class string) float64 {
	// Recall Bayes Theorem:
	// P(A | B) = P(A) * P(B | A) / P(B)
	// P(pos | words) = P(pos) * P(words | pos) / P(words)

	// We can drop the constant denominator for implementation purposes.
	// We also apply logarithmic transform to help bias toward zero

	prob := math.Log(c.prior(class))
	for _, w := range words {
		count := 0
		if wf, ok := c.words[w]; ok {
			count = wf.counter[class]
		}
		prob += math.Log(float64(count + alpha) / float64(c.totalWordCount(class) + 2*alpha))
	}
	return prob
}
