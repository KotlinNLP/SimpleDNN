/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package mnist

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.GRUNeuralNetwork
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.dataset.*
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.ClassificationEvaluation
import com.kotlinnlp.simplednn.simplemath.*
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.helpers.training.SequenceWithFinalOutputTrainingHelper
import com.kotlinnlp.simplednn.helpers.validation.SequenceWithFinalOutputValidationHelper
import com.jsoniter.*
import Configuration
import CorpusPaths
import java.io.File
import java.util.concurrent.TimeUnit

fun main(args: Array<String>) {
  println("Start 'MNIST Sequence Test'")
  MNISTSequenceTest.start()
  println("End.")
}

/**
 *
 */
object MNISTSequenceTest {

  /**
   *
   */
  fun start(): Unit {

    val configuration = Configuration.loadFromFile()
    val dataset: Corpus<SequenceExampleWithFinalOutput> = this.readCorpus(configuration.mnist_sequence.datasets_paths)

    val nn = this.buildNetwork()

    this.train(nn, dataset)
  }

  /**
   *
   * @param corpus corpus
   * @return
   */
  fun readCorpus(corpus: CorpusPaths): Corpus<SequenceExampleWithFinalOutput> {

    println("\n-- CORPUS READING")

    val startTime = System.nanoTime()

    val dataset = Corpus(
      training = JSONLFileReader(corpus.training),
      validation = JSONLFileReader(corpus.validation),
      test = JSONLFileReader(corpus.test))

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
  fun JSONLFileReader(filename: String): ArrayList<SequenceExampleWithFinalOutput> {

    val file = File(filename)
    val examples: ArrayList<SequenceExampleWithFinalOutput> = ArrayList()

    file.forEachLine {
      examples.add(this.extractExample(iterator = JsonIterator.parse(it)))
    }

    return examples
  }

  /**
   *
   */
  fun extractExample(iterator: JsonIterator): SequenceExampleWithFinalOutput {

    val featuresList = ArrayList<NDArray>()
    val outputGold = NDArray.zeros(Shape(10))

    // read "digit"
    iterator.readObject()
    outputGold[iterator.readInt()] = 1.0

    // skip "id"
    iterator.readObject()
    iterator.readAny()

    // read "sequence_data"
    iterator.readObject()

    while (iterator.readArray()) {
      if (iterator.whatIsNext() == ValueType.ARRAY) {
        val features = iterator.readNDArray()
        val deltaX = features[0].toDouble()
        val deltaY = features[1].toDouble()
        featuresList.add(NDArray.arrayOf(doubleArrayOf(deltaX, deltaY)))
      }
    }

    return SequenceExampleWithFinalOutput(featuresList, outputGold)
  }

  /**
   *
   */
  fun buildNetwork(): NeuralNetwork {

    val nn = GRUNeuralNetwork(
      inputSize = 2,
      hiddenSize = 200,
      hiddenActivation = Tanh(),
      outputSize = 10,
      outputActivation = Softmax()
    )

    nn.initialize()

    return nn
  }

  /**
   *
   */
  fun train(neuralNetwork: NeuralNetwork, dataset: Corpus<SequenceExampleWithFinalOutput>): Unit {

    println("\n-- TRAINING")

    val optimizer = ParamsOptimizer(neuralNetwork, ADAMMethod(stepSize = 0.001))

    val neuralProcessor = RecurrentNeuralProcessor(neuralNetwork)

    val trainingHelper = SequenceWithFinalOutputTrainingHelper(
      neuralProcessor = neuralProcessor,
      optimizer = optimizer,
      lossCalculator = MSECalculator(),
      verbose = true)

    val validationHelper = SequenceWithFinalOutputValidationHelper(
      neuralProcessor = neuralProcessor,
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
