package main

import(
	"fmt"
	"math"
	//cloud "github.com/ryanbressler/CloudForest"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)



func Sqrt(x float64) float64 {
	z := x/2
	i := 1
	for {
		z -= (z*z - x) / (2*z)
		fmt.Println("At iteration", i , "the square root is equal to", z)
		i++
		if j := z - math.Sqrt(x); j <= 0.0000000000000 {
			break
		}
	}

	return z
}
func preprocessing (datapath string) *base.DenseInstances {
	data, err := base.ParseCSVToInstances("invdataset.csv", true)
	if err != nil {
		panic(err)
	}
	return data
}
func main() {




	// Load in the iris dataset
	data, err := base.ParseCSVToInstances("invdataset.csv", true)
	if err != nil {
		panic(err)
	}
	datatest, err := base.ParseCSVToInstances("invdatatree.csv", true)
	if err != nil {
		panic(err)
	}

	//Initialises a new KNN classifier
	cls := knn.NewKnnClassifier("euclidean", "kdtree", 2)

	//Do a training-test split
	
	cls.Fit(data)

	//Calculates the Euclidean distance and returns the most popular label
	predictions, err := cls.Predict(datatest)
	if err != nil {
		panic(err)
	}
	fmt.Println(predictions)

	// Prints precision/recall metrics
	confusionMat, err := evaluation.GetConfusionMatrix(datatest, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(confusionMat))
}
/*	a,b := cloud.LoadAFM("/home/marco/Softcompexam/data.afm")
	var cfeat cloud.CatFeature = a.Data[0]
	fmt.Println(cfeat)
	tree := cloud.NewTree()

	tree.Grow(a)
	fmt.Println("\n",b)
}*/