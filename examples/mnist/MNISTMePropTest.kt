/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package mnist

import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.FeedforwardNeuralNetwork
import com.kotlinnlp.simplednn.helpers.training.FeedforwardTrainingHelper
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.dataset.*
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.ClassificationEvaluation
import com.kotlinnlp.simplednn.helpers.validation.FeedforwardValidationHelper
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import utils.exampleextractor.ClassificationExampleExtractor
import utils.CorpusReader
import Configuration
import com.kotlinnlp.simplednn.core.functionalities.activations.ReLU
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod

fun main(args: Array<String>) {

  println("Start 'MNIST meProp Test'")

  val dataset = CorpusReader<SimpleExample<DenseNDArray>>().read(
    corpusPath = Configuration.loadFromFile().mnist.datasets_paths,
    exampleExtractor = ClassificationExampleExtractor(outputSize = 10),
    perLine = false)

  MNISTMePropTest(dataset).start()

  println("\nEnd.")
}

/**
 *
 */
class MNISTMePropTest(val dataset: Corpus<SimpleExample<DenseNDArray>>) {

  /**
   *
   */
  private val neuralNetwork = FeedforwardNeuralNetwork(
    inputSize = 784,
    hiddenSize = 500,
    hiddenActivation = ReLU(),
    hiddenMeProp = true,
    outputSize = 10,
    outputActivation = Softmax())

  /**
   *
   */
  fun start() {
    this.train()
    this.evaluate()
  }

  /**
   *
   */
  private fun train() {

    println("\n-- TRAINING")

    val optimizer = ParamsOptimizer(
      params = this.neuralNetwork.model,
      updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999))

    val trainingHelper = FeedforwardTrainingHelper<DenseNDArray>(
      neuralProcessor = FeedforwardNeuralProcessor(
        neuralNetwork = this.neuralNetwork,
        useDropout = false,
        propagateToInput = false,
        mePropK = listOf(0.16, null)),
      optimizer = optimizer,
      lossCalculator = SoftmaxCrossEntropyCalculator(),
      verbose = true)

    val validationHelper = FeedforwardValidationHelper<DenseNDArray>(
      neuralProcessor = FeedforwardNeuralProcessor(this.neuralNetwork, propagateToInput = false, useDropout = false),
      outputEvaluationFunction = ClassificationEvaluation())

    trainingHelper.train(
      trainingExamples = this.dataset.training,
      validationExamples = this.dataset.validation,
      epochs = 15,
      batchSize = 10,
      shuffler = Shuffler(enablePseudoRandom = true, seed = 1),
      validationHelper = validationHelper)
  }

  /**
   *
   */
  private fun evaluate() {

    val validationHelper = FeedforwardValidationHelper<DenseNDArray>(
      neuralProcessor = FeedforwardNeuralProcessor(this.neuralNetwork, propagateToInput = false, useDropout = false),
      outputEvaluationFunction = ClassificationEvaluation())

    println("\n-- EVALUATION ON %d TEST SENTENCES".format(this.dataset.test.size))

    val accuracy: Double = validationHelper.validate(this.dataset.test)

    println("Accuracy: %.2f%%".format(100.0 * accuracy))
  }
}
