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
import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.simplemath.ndarray.*
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import mnist.helpers.MNISTSparseExampleExtractor
import CorpusReader

fun main(args: Array<String>) {

  println("Start 'MNIST Sparse Binary Test'")

  val dataset = CorpusReader<SimpleExample<SparseBinaryNDArray>>().read(
    corpusPath = Configuration.loadFromFile().mnist.datasets_paths,
    exampleExtractor = MNISTSparseExampleExtractor(outputSize = 10),
    perLine = false)

  MNISTSparseBinaryTest(dataset).start()

  println("End.")
}

/**
 *
 */
class MNISTSparseBinaryTest(val dataset: Corpus<SimpleExample<SparseBinaryNDArray>>) {

  /**
   *
   */
  private val neuralNetwork = this.buildNetwork()

  /**
   *
   */
  fun start(): Unit {

    this.train()
    this.printImages(examples = ArrayList(this.dataset.validation.subList(0, 20))) // reduced sublist
  }

  /**
   *
   */
  private fun buildNetwork(): NeuralNetwork {

    val nn = FeedforwardNeuralNetwork(
      inputSize = 784,
      inputType = LayerType.Input.SparseBinary,
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

    println("\n-- TRAINING\n")

    val optimizer = ParamsOptimizer(
      neuralNetwork = this.neuralNetwork,
      updateMethod = LearningRateMethod(
        learningRate = 0.01,
        decayMethod = HyperbolicDecay(decay = 0.5, initLearningRate = 0.01)))

    val trainingHelper = FeedforwardTrainingHelper<SparseBinaryNDArray>(
      neuralProcessor = FeedforwardNeuralProcessor(this.neuralNetwork),
      optimizer = optimizer,
      lossCalculator = SoftmaxCrossEntropyCalculator(),
      verbose = true)

    val validationHelper = FeedforwardValidationHelper<SparseBinaryNDArray>(
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

  /**
   *
   */
  private fun printImages(examples: ArrayList<SimpleExample<SparseBinaryNDArray>>) {

    println("\n-- PRINT IMAGES RELEVANCE\n")

    val neuralProcessor = FeedforwardNeuralProcessor<SparseBinaryNDArray>(neuralNetwork)

    val validationHelper = FeedforwardValidationHelper(
      neuralProcessor = neuralProcessor,
      outputEvaluationFunction = ClassificationEvaluation())

    validationHelper.validate(
      examples = examples,
      saveContributions = true,
      onPrediction = { example, _ ->
        val sparseRelevance = neuralProcessor.calculateInputRelevance(DistributionArray.uniform(length = 10))
        val denseRelevance: DenseNDArray = DenseNDArrayFactory.zeros(Shape(784)).assignValues(sparseRelevance)

        this.printImage(image = denseRelevance, value = example.outputGold.argMaxIndex())
      }
    )
  }

  /**
   *
   */
  private fun printImage(image: DenseNDArray, value: Int) {

    println("------------------ %d -----------------".format(value))

    for (i in 0 until 28) {
      for (j in 0 until 28) {
          print(if (image[i * 28 + j] > 0.0) "# " else "  ")
      }
      println()
    }
  }
}
