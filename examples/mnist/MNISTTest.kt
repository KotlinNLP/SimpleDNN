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
import com.kotlinnlp.simplednn.simplemath.NDArray
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.FeedforwardNeuralNetwork
import com.kotlinnlp.simplednn.helpers.training.FeedforwardTrainingHelper
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.dataset.*
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.ClassificationEvaluation
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.helpers.validation.FeedforwardValidationHelper
import com.jsoniter.*
import Configuration
import CorpusPaths
import java.io.BufferedInputStream
import java.io.FileInputStream
import java.util.concurrent.TimeUnit

fun main(args: Array<String>) {
  println("Start 'MNIST Test'")
  MNISTTest.start()
  println("End.")
}


/**
 *
 */
object MNISTTest {

  /**
   *
   */
  fun start(): Unit {

    val configuration = Configuration.Companion.loadFromFile()
    val dataset: Corpus<SimpleExample> = readCorpus(configuration.mnist.datasets_paths)

    val nn = buildNetwork()

    train(nn, dataset)
  }

  /**
   *
   * @param corpus corpus
   * @return
   */
  fun readCorpus(corpus: CorpusPaths): Corpus<SimpleExample> {

    println("\n-- CORPUS READING")

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
  fun JsonIterator.readNDArray(): NDArray {
    val array = ArrayList<Double>()
    while (this.readArray()) array.add(this.readDouble())
    return NDArray.arrayOf(array.toDoubleArray())
  }

  /**
   *
   * @param filename
   * @return
   */
  fun JSONFileReader(filename: String): ArrayList<SimpleExample> {

    val examples: ArrayList<SimpleExample> = ArrayList()

    val iter: JsonIterator = JsonIterator.parse(BufferedInputStream(FileInputStream(filename)), 2048)

    // loop samples
    while(iter.readArray()) {

      val outputGold = NDArray.zeros(Shape(1, 10))

      var goldIndex: Int

      var features: NDArray? = null

      // Single Example
      while (iter.readArray()) {
        if (iter.whatIsNext() == ValueType.ARRAY) {
          features = iter.readNDArray()
        } else if (iter.whatIsNext() == ValueType.NUMBER) {
          goldIndex = iter.readInt() // - 1
          outputGold[goldIndex] = 1.0
        }
      }

      val example = SimpleExample(features!!, outputGold)

      examples.add(example)

    }

    return examples
  }

  /**
   *
   */
  fun buildNetwork(): NeuralNetwork {

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
  fun train(neuralNetwork: NeuralNetwork, dataset: Corpus<SimpleExample>): Unit {

    println("\n-- TRAINING")

    val optimizer = ParamsOptimizer(
      neuralNetwork = neuralNetwork,
      updateMethod = LearningRateMethod(
        learningRate = 0.01,
        decayMethod = HyperbolicDecay(decay = 0.5, initLearningRate = 0.01)))

    val trainingHelper = FeedforwardTrainingHelper(
      neuralProcessor = FeedforwardNeuralProcessor(neuralNetwork),
      optimizer = optimizer,
      lossCalculator = MSECalculator(),
      verbose = true)

    val validationHelper = FeedforwardValidationHelper(
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
}
