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
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.FeedforwardNeuralNetwork
import com.kotlinnlp.simplednn.helpers.training.FeedforwardTrainingHelper
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.dataset.*
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.ClassificationEvaluation
import com.kotlinnlp.simplednn.helpers.validation.FeedforwardValidationHelper
import com.jsoniter.*
import Configuration
import CorpusPaths
import com.kotlinnlp.simplednn.core.arrays.DistributionArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.simplemath.ndarray.*
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArrayFactory
import java.io.BufferedInputStream
import java.io.FileInputStream
import java.util.concurrent.TimeUnit

fun main(args: Array<String>) {
  println("Start 'MNIST Sparse Binary Test'")
  MNISTSparseBinaryTest.start()
  println("End.")
}

/**
 *
 */
object MNISTSparseBinaryTest {

  /**
   *
   */
  fun start(): Unit {

    val configuration = Configuration.Companion.loadFromFile()
    val dataset: Corpus<SimpleExample<SparseBinaryNDArray>> = readCorpus(configuration.mnist.datasets_paths)

    val nn = buildNetwork()

    train(neuralNetwork = nn, dataset = dataset)

    printImages(
      neuralNetwork = nn,
      dataset = arrayListOf(*dataset.validation.subList(0, 20).toTypedArray()) // reduced sublist
    )
  }

  /**
   *
   */
  fun readCorpus(corpus: CorpusPaths): Corpus<SimpleExample<SparseBinaryNDArray>> {

    println("\n-- CORPUS READING\n")

    val startTime = System.nanoTime()

    val dataset = Corpus(
      training = JSONFileReader(corpus.training),
      validation = JSONFileReader(corpus.validation),
      test = JSONFileReader(corpus.test))

    val elapsedTime = System.nanoTime() - startTime

    println("Elapsed time: ${TimeUnit.MILLISECONDS.convert(elapsedTime, TimeUnit.NANOSECONDS) / 1000.0}s")
    println("Train: %d examples".format(dataset.training.size))
    println("Validation: %d examples".format(dataset.validation.size))
    println("Test: %d examples".format(dataset.test.size))

    return dataset
  }

  /**
   *
   */
  fun JsonIterator.readNDArray(): SparseBinaryNDArray {

    val array = ArrayList<Int>()
    var index: Int = 0

    while (this.readArray()) {

      if (Math.round(this.readDouble()).toInt() == 1) {
        array.add(index)
      }

      index++
    }

    return SparseBinaryNDArrayFactory.arrayOf(activeIndices = array.toIntArray(), shape = Shape(784))
  }

  /**
   *
   */
  fun JSONFileReader(filename: String): ArrayList<SimpleExample<SparseBinaryNDArray>> {

    val examples = ArrayList<SimpleExample<SparseBinaryNDArray>>()
    val iterator = JsonIterator.parse(BufferedInputStream(FileInputStream(filename)), 2048)

    while(iterator.readArray()) {
      examples.add(this.extractExample(iterator))
    }

    return examples
  }

  /**
   *
   */
  fun extractExample(iterator: JsonIterator): SimpleExample<SparseBinaryNDArray> {

    val outputGold = DenseNDArrayFactory.zeros(Shape(1, 10))
    var goldIndex: Int
    var features: SparseBinaryNDArray? = null

    while (iterator.readArray()) {

      if (iterator.whatIsNext() == ValueType.ARRAY) {
        features = iterator.readNDArray()

      } else if (iterator.whatIsNext() == ValueType.NUMBER) {
        goldIndex = iterator.readInt() // - 1
        outputGold[goldIndex] = 1.0
      }
    }

    return SimpleExample(features!!, outputGold)
  }

  /**
   *
   */
  fun buildNetwork(): NeuralNetwork {

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
  fun train(neuralNetwork: NeuralNetwork, dataset: Corpus<SimpleExample<SparseBinaryNDArray>>): Unit {

    println("\n-- TRAINING\n")

    val optimizer = ParamsOptimizer(
      neuralNetwork = neuralNetwork,
      updateMethod = LearningRateMethod(
        learningRate = 0.01,
        decayMethod = HyperbolicDecay(decay = 0.5, initLearningRate = 0.01)))

    val trainingHelper = FeedforwardTrainingHelper<SparseBinaryNDArray>(
      neuralProcessor = FeedforwardNeuralProcessor(neuralNetwork),
      optimizer = optimizer,
      lossCalculator = MSECalculator(),
      verbose = true)

    val validationHelper = FeedforwardValidationHelper<SparseBinaryNDArray>(
      neuralProcessor = FeedforwardNeuralProcessor(neuralNetwork),
      outputEvaluationFunction = ClassificationEvaluation())

    trainingHelper.train(
      trainingExamples = dataset.training,
      validationExamples = dataset.validation,
      epochs = 3,
      batchSize = 1,
      shuffler = Shuffler(enablePseudoRandom = true, seed = 1),
      validationHelper = validationHelper)
  }

  /**
   *
   */
  fun printImages(neuralNetwork: NeuralNetwork, dataset: ArrayList<SimpleExample<SparseBinaryNDArray>>) {

    println("\n-- PRINT IMAGES RELEVANCE\n")

    val neuralProcessor = FeedforwardNeuralProcessor<SparseBinaryNDArray>(neuralNetwork)

    val validationHelper = FeedforwardValidationHelper(
      neuralProcessor = neuralProcessor,
      outputEvaluationFunction = ClassificationEvaluation())

    validationHelper.validate(
      examples = dataset,
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
  fun printImage(image: DenseNDArray, value: Int) {

    println("------------------ %d -----------------".format(value))

    for (i in 0 until 28) {
      for (j in 0 until 28) {
          print(if (image[i * 28 + j] > 0.0) "# " else "  ")
      }
      println()
    }
  }
}
