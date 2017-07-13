/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.dataset.*
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.ClassificationEvaluation
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.SimpleRecurrentNeuralNetwork
import com.kotlinnlp.simplednn.helpers.training.SequenceWithFinalOutputTrainingHelper
import com.kotlinnlp.simplednn.helpers.validation.SequenceWithFinalOutputValidationHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

fun main(args: Array<String>) {
  println("Start 'Sum Sign Relevance Test'")
  SumSignRelevanceTest.start()
  println("End.")
}

/**
 *
 */
object SumSignRelevanceTest {

  /**
   * The number of examples to print.
   */
  val examplesPrintCount: Int = 20

  /**
   * Start the test.
   */
  fun start() {

    val dataset = this.getDataset(size = 10000)
    val nn = this.buildNetwork()

    train(neuralNetwork = nn, dataset = dataset)
    printRelevance(neuralNetwork = nn, testset = this.getDataset(size = 100))
  }

  /**
   * @return a new neural network
   */
  fun buildNetwork(): NeuralNetwork {

    val nn = SimpleRecurrentNeuralNetwork(
      inputSize = 1,
      hiddenSize = 10,
      hiddenActivation = Tanh(),
      outputSize = 3,
      outputActivation = Softmax())

    nn.initialize(biasesInitValue = 0.0)

    return nn
  }

  /**
   * @param size the size of the dataset
   *
   * @return a dataset of examples of sequences
   */
  fun getDataset(size: Int): ArrayList<SequenceExampleWithFinalOutput<DenseNDArray>> =
    arrayListOf(*Array(size = size, init = { this.createExample() }))

  /**
   * @return an example containing a sequence of single features with a random value in {-1.0, 0.0, 1.0} and a gold
   *         output which is the sign of the sum of the features (represented by a one hot encoder [0 = negative,
   *         1 = zero sign, 2 = positive])
   */
  fun createExample(): SequenceExampleWithFinalOutput<DenseNDArray> {

    val features = arrayListOf(*Array(size = 10, init = { this.getRandomInput() }))
    val outputGoldIndex = Math.signum(features.sumByDouble { it[0] }) + 1

    return SequenceExampleWithFinalOutput(
      sequenceFeatures = features,
      outputGold = DenseNDArrayFactory.oneHotEncoder(length = 3, oneAt = outputGoldIndex.toInt())
    )
  }

  /**
   * @return a [DenseNDArray] containing a single value within {-1.0, 0.0, 1.0}
   */
  fun getRandomInput(): DenseNDArray {
      val value = Math.round(Math.random() * 2.0 - 1.0).toDouble()
      return DenseNDArrayFactory.arrayOf(doubleArrayOf(value))
  }

  /**
   * Train the network on a dataset.
   *
   * @param neuralNetwork a neural network
   * @param dataset the dataset used to train the network
   */
  fun train(neuralNetwork: NeuralNetwork, dataset: ArrayList<SequenceExampleWithFinalOutput<DenseNDArray>>) {

    println("\n-- TRAINING\n")

    val optimizer = ParamsOptimizer(
      neuralNetwork = neuralNetwork,
      updateMethod = LearningRateMethod(learningRate = 0.01))

    val trainingHelper = SequenceWithFinalOutputTrainingHelper<DenseNDArray>(
      neuralProcessor = RecurrentNeuralProcessor(neuralNetwork),
      optimizer = optimizer,
      lossCalculator = SoftmaxCrossEntropyCalculator(),
      verbose = true)

    val validationHelper = SequenceWithFinalOutputValidationHelper(
      neuralProcessor = RecurrentNeuralProcessor<DenseNDArray>(neuralNetwork),
      outputEvaluationFunction = ClassificationEvaluation())

    trainingHelper.train(
      trainingExamples = dataset,
      validationExamples = arrayListOf(*dataset.subList(0, 1000).toTypedArray()), // reduced sublist
      epochs = 3,
      shuffler = Shuffler(enablePseudoRandom = true, seed = 1),
      batchSize = 1,
      validationHelper = validationHelper)
  }

  /**
   * Print the relevance of each example of the dataset.
   *
   * @param neuralNetwork a neural network
   * @param testset the test dataset
   */
  fun printRelevance(neuralNetwork: NeuralNetwork,
                     testset: ArrayList<SequenceExampleWithFinalOutput<DenseNDArray>>) {

    println("\n-- PRINT RELEVANCE OF %d EXAMPLES\n".format(this.examplesPrintCount))

    val validationProcessor = RecurrentNeuralProcessor<DenseNDArray>(neuralNetwork)

    val validationHelper = SequenceWithFinalOutputValidationHelper(
      neuralProcessor = validationProcessor,
      outputEvaluationFunction = ClassificationEvaluation())

    var exampleIndex = 0

    validationHelper.validate(testset,
      saveContributions = true,
      onPrediction = { example, isCorrect ->
        if (isCorrect && exampleIndex < this.examplesPrintCount) {
          this.printSequenceRelevance(
            neuralProcessor = validationProcessor,
            example = example,
            exampleIndex = exampleIndex++)
        }
      })
  }

  /**
   * Print the relevance of each input of the sequence.
   *
   * @param neuralProcessor the neural processor of the validation
   * @param example the validated sequence
   */
  fun printSequenceRelevance(neuralProcessor: RecurrentNeuralProcessor<DenseNDArray>,
                             example: SequenceExampleWithFinalOutput<DenseNDArray>,
                             exampleIndex: Int) {

    val sequenceRelevance = this.getSequenceRelevance(
      neuralProcessor = neuralProcessor,
      outputGold = example.outputGold
    )

    println("EXAMPLE %d".format(exampleIndex + 1))
    println("Gold: %d".format(example.outputGold.argMaxIndex() - 1))
    println("Sequence (input | relevance):")

    (0 until sequenceRelevance.size).forEach { i ->
      println("\t%4.1f | %8.1f".format(example.sequenceFeatures[i][0], sequenceRelevance[i]))
    }

    println()
  }

  /**
   * @param neuralProcessor the neural processor of the validation
   * @param outputGold the gold output array
   *
   * @return an array containing the relevance for each input of the sequence in respect of the gold output
   */
  fun getSequenceRelevance(neuralProcessor: RecurrentNeuralProcessor<DenseNDArray>,
                           outputGold: DenseNDArray): Array<Double> {

    val outcomesDistr = DistributionArray.oneHot(length = 3, oneAt = outputGold.argMaxIndex())

    return Array(
      size = 10,
      init = { i ->
        neuralProcessor.calculateRelevance(
          stateFrom = i,
          stateTo = 9,
          relevantOutcomesDistribution = outcomesDistr).sum()
      }
    )
  }
}
