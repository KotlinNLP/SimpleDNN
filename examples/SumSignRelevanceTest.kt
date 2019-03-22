/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.ClassificationEvaluation
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.SimpleRecurrentNeuralNetwork
import com.kotlinnlp.simplednn.core.optimizer.GenericParamsOptimizer
import traininghelpers.training.SequenceWithFinalOutputTrainingHelper
import traininghelpers.validation.SequenceWithFinalOutputValidationHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.Shuffler
import utils.Corpus
import utils.SequenceExampleWithFinalOutput

fun main(args: Array<String>) {

  println("Start 'Sum Sign Relevance Test'")

  SumSignRelevanceTest(dataset = DatasetBuilder.build()).start()

  println("End.")
}

/**
 *
 */
object DatasetBuilder {

  /**
   * @return a dataset of examples of sequences
   */
  fun build(): Corpus<SequenceExampleWithFinalOutput<DenseNDArray>> = Corpus(
    training = arrayListOf(*Array(size = 10000, init = { this.createExample() })),
    validation = arrayListOf(*Array(size = 1000, init = { this.createExample() })),
    test = arrayListOf(*Array(size = 100, init = { this.createExample() }))
  )

  /**
   * @return an example containing a sequence of single features with a random value in {-1.0, 0.0, 1.0} and a gold
   *         output which is the sign of the sum of the features (represented by a one hot encoder [0 = negative,
   *         1 = zero sign, 2 = positive])
   */
  private fun createExample(): SequenceExampleWithFinalOutput<DenseNDArray> {

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
  private fun getRandomInput(): DenseNDArray {
    val value = Math.round(Math.random() * 2.0 - 1.0).toDouble()
    return DenseNDArrayFactory.arrayOf(doubleArrayOf(value))
  }
}

/**
 *
 */
class SumSignRelevanceTest(val dataset: Corpus<SequenceExampleWithFinalOutput<DenseNDArray>>) {

  /**
   * The number of examples to print.
   */
  private val examplesToPrint: Int = 20

  /**
   *
   */
  private val neuralNetwork = SimpleRecurrentNeuralNetwork(
    inputSize = 1,
    hiddenSize = 10,
    hiddenActivation = Tanh(),
    outputSize = 3,
    outputActivation = Softmax())

  /**
   * Start the test.
   */
  fun start() {

    this.train()
    this.printRelevance()
  }

  /**
   * Train the network on a dataset.
   */
  private fun train() {

    println("\n-- TRAINING\n")

    val optimizer = GenericParamsOptimizer(
      updateMethod = LearningRateMethod(learningRate = 0.01))

    val trainingHelper = SequenceWithFinalOutputTrainingHelper<DenseNDArray>(
      neuralProcessor = RecurrentNeuralProcessor(
        model = this.neuralNetwork,
        useDropout = false,
        propagateToInput = false),
      optimizer = optimizer,
      lossCalculator = SoftmaxCrossEntropyCalculator(),
      verbose = true)

    val validationHelper = SequenceWithFinalOutputValidationHelper(
      neuralProcessor = RecurrentNeuralProcessor<DenseNDArray>(
        model = this.neuralNetwork,
        useDropout = false,
        propagateToInput = false),
      outputEvaluationFunction = ClassificationEvaluation())

    trainingHelper.train(
      trainingExamples = this.dataset.training,
      validationExamples = this.dataset.validation,
      epochs = 3,
      shuffler = Shuffler(enablePseudoRandom = true, seed = 1),
      batchSize = 1,
      validationHelper = validationHelper)
  }

  /**
   * Print the relevance of each example of the dataset.
   */
  private fun printRelevance() {

    println("\n-- PRINT RELEVANCE OF %d EXAMPLES\n".format(this.examplesToPrint))

    val validationProcessor = RecurrentNeuralProcessor<DenseNDArray>(
      model = neuralNetwork,
      useDropout = false,
      propagateToInput = false)

    val validationHelper = SequenceWithFinalOutputValidationHelper(
      neuralProcessor = validationProcessor,
      outputEvaluationFunction = ClassificationEvaluation())

    var exampleIndex = 0

    validationHelper.validate(
      examples = this.dataset.test,
      saveContributions = true,
      onPrediction = { example, isCorrect ->
        if (isCorrect && exampleIndex < this.examplesToPrint) {
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
  private fun printSequenceRelevance(neuralProcessor: RecurrentNeuralProcessor<DenseNDArray>,
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
  private fun getSequenceRelevance(neuralProcessor: RecurrentNeuralProcessor<DenseNDArray>,
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
