/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateMethod
import com.kotlinnlp.simplednn.simplemath.NDArray
import com.jsoniter.*
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.CFNNeuralNetwork
import com.kotlinnlp.simplednn.helpers.training.SequenceTrainingHelper
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.dataset.*
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.ClassificationEvaluation
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.helpers.validation.SequenceValidationHelper
import java.io.File
import java.util.concurrent.TimeUnit

fun main(args: Array<String>) {
  println("Start 'Progressive Sum Test'")
  ProgressiveSumTest.start()
  println("End.")
}

/**
 *
 */
object ProgressiveSumTest {

  /**
   *
   */
  fun start(): Unit {

    val configuration = Configuration.loadFromFile()
    val dataset: Corpus<SequenceExample> = this.readCorpus(configuration.progressive_sum.datasets_paths)

    val nn = this.buildNetwork()

    this.initialValidation(nn, dataset)

    this.train(nn, dataset)
  }

  /**
   *
   * @param corpus corpus
   * @return
   */
  fun readCorpus(corpus: CorpusPaths): Corpus<SequenceExample> {

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
  fun JSONLFileReader(filename: String): ArrayList<SequenceExample> {

    val file = File(filename)
    val examples = ArrayList<SequenceExample>()

    file.forEachLine {
      val iterator: JsonIterator = JsonIterator.parse(it)
      val (featuresList, outputGoldList) = this.extractExampleData(iterator)

      examples.add(SequenceExample(featuresList, outputGoldList))
    }

    return examples
  }

  /**
   *
   */
  fun extractExampleData(iterator: JsonIterator): Pair<ArrayList<NDArray>, ArrayList<NDArray>> {

    val featuresList = ArrayList<NDArray>()
    val outputGoldList = ArrayList<NDArray>()

    while (iterator.readArray()) {
      if (iterator.whatIsNext() == ValueType.ARRAY) {
        val singleExample = iterator.readNDArray()
        val features = NDArray.arrayOf(doubleArrayOf(singleExample[0].toDouble()))
        val outputGold = NDArray.zeros(Shape(11))

        outputGold[singleExample[1].toInt()] = 1.0

        featuresList.add(features)
        outputGoldList.add(outputGold)
      }
    }

    return Pair(featuresList, outputGoldList)
  }

  /**
   *
   */
  fun buildNetwork(): NeuralNetwork {

    val nn = CFNNeuralNetwork(
      inputSize = 1,
      hiddenSize = 100,
      hiddenActivation = Tanh(),
      outputSize = 11,
      outputActivation = Softmax())

    nn.initialize(biasesInitValue = 0.0)

    return nn
  }

  /**
   *
   */
  fun initialValidation(neuralNetwork: NeuralNetwork, dataset: Corpus<SequenceExample>): Unit {

    println("\n-- VALIDATION BEFORE TRAINING")

    val validationHelper = SequenceValidationHelper(
      neuralProcessor = RecurrentNeuralProcessor(neuralNetwork),
      outputEvaluationFunction = ClassificationEvaluation())

    val accuracy: Double = validationHelper.validate(dataset.validation)

    println("Accuracy: %.2f%%".format(100.0 * accuracy))
  }

  /**
   *
   */
  fun train(neuralNetwork: NeuralNetwork, dataset: Corpus<SequenceExample>): Unit {

    println("\n-- TRAINING")

    val optimizer = ParamsOptimizer(
      neuralNetwork = neuralNetwork,
      updateMethod = LearningRateMethod(learningRate = 0.1))

    val trainingHelper = SequenceTrainingHelper(
      neuralProcessor = RecurrentNeuralProcessor(neuralNetwork),
      optimizer = optimizer,
      lossCalculator = MSECalculator(),
      verbose = true)

    val validationHelper = SequenceValidationHelper(
      neuralProcessor = RecurrentNeuralProcessor(neuralNetwork),
      outputEvaluationFunction = ClassificationEvaluation())

    trainingHelper.train(
      trainingExamples = dataset.training,
      validationExamples = dataset.validation,
      epochs = 4,
      shuffler = Shuffler(enablePseudoRandom = true, seed = 1),
      batchSize = 1,
      validationHelper = validationHelper)
  }
}
