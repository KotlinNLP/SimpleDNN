/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package mnist

import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.core.functionalities.activations.ELU
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.decaymethods.HyperbolicDecay
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateMethod
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.FeedforwardNeuralNetwork
import com.kotlinnlp.simplednn.helpers.training.FeedforwardTrainingHelper
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.dataset.*
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.ClassificationEvaluation
import com.kotlinnlp.simplednn.helpers.validation.FeedforwardValidationHelper
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import Configuration
import CorpusReader
import ClassificationExampleExtractor

fun main(args: Array<String>) {

  println("Start 'MNIST Test'")

  val dataset = CorpusReader<SimpleExample<DenseNDArray>>().read(
    corpusPath = Configuration.loadFromFile().mnist.datasets_paths,
    exampleExtractor = ClassificationExampleExtractor(outputSize = 10),
    perLine = false)

  MNISTTest(dataset).start()

  println("End.")
}

/**
 *
 */
class MNISTTest(val dataset: Corpus<SimpleExample<DenseNDArray>>) {

  /**
   *
   */
  private val neuralNetwork = this.buildNetwork()

  /**
   *
   */
  fun start(): Unit {
    this.train()
  }

  /**
   *
   */
  private fun buildNetwork(): NeuralNetwork {

    val nn = FeedforwardNeuralNetwork(
      inputSize = 784,
      hiddenSize = 100,
      hiddenActivation = ELU(),
      outputSize = 10,
      outputActivation = Softmax())

    nn.initialize()

    return nn
  }

  /**
   *
   */
  private fun train(): Unit {

    println("\n-- TRAINING")

    val optimizer = ParamsOptimizer(
      neuralNetwork = this.neuralNetwork,
      updateMethod = LearningRateMethod(
        learningRate = 0.01,
        decayMethod = HyperbolicDecay(decay = 0.5, initLearningRate = 0.01)))

    val trainingHelper = FeedforwardTrainingHelper<DenseNDArray>(
      neuralProcessor = FeedforwardNeuralProcessor(this.neuralNetwork),
      optimizer = optimizer,
      lossCalculator = SoftmaxCrossEntropyCalculator(),
      verbose = true)

    val validationHelper = FeedforwardValidationHelper<DenseNDArray>(
      neuralProcessor = FeedforwardNeuralProcessor(this.neuralNetwork),
      outputEvaluationFunction = ClassificationEvaluation())

    trainingHelper.train(
      trainingExamples = this.dataset.training,
      validationExamples = this.dataset.validation,
      epochs = 3,
      batchSize = 1,
      shuffler = Shuffler(enablePseudoRandom = true, seed = 1),
      validationHelper = validationHelper)
  }
}
