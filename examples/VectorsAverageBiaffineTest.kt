/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam.RADAMMethod
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.utils.ShuffledIndices
import com.kotlinnlp.utils.Shuffler
import java.io.File
import kotlin.math.roundToInt

fun main() {

  println("Start 'Vectors Average Biaffine Test'")

  val corpusPath = Configuration.loadFromFile().vectors_average.datasets_paths

  VectorsAverageBiaffineTest(corpusPath.training).start()

  println("\nEnd.")
}

typealias Example = Triple<DenseNDArray, DenseNDArray, DenseNDArray>

/**
 *
 */
class VectorsAverageBiaffineTest(private val trainingSetPath: String) {

  /**
   *
   */
  private val shuffler = Shuffler()

  /**
   *
   */
  private val biaffineProcessor = FeedforwardNeuralProcessor<DenseNDArray>(
    model = StackedLayersParameters(
      LayerInterface(sizes = listOf(5, 5)),
      LayerInterface(size = 5, connectionType = LayerType.Connection.Biaffine)),
    propagateToInput = false
  )

  /**
   *
   */
  private val optimizer = ParamsOptimizer(
    updateMethod = RADAMMethod(stepSize = 0.001, beta1 = 0.99, beta2 = 0.99999))

  /**
   *
   */
  fun start() {

    val dataset: ArrayList<Example> = this.loadExamples()
    val testSetSize: Int = (0.1 * dataset.size).roundToInt()
    val testSet = dataset.subList(fromIndex = 0, toIndex = testSetSize)
    val trainingSet = dataset.subList(fromIndex = testSetSize, toIndex = dataset.size)

    val epochs = 25

    println("\n-- TRAINING ON ${trainingSet.size} EXAMPLES")

    (0 until epochs).forEach { epoch ->

      println("\nEpoch ${epoch + 1} of $epochs")

      this.trainEpoch(trainingExamples = trainingSet)

      println("\nValidation on ${testSet.size} examples")
      println("Accuracy: %.2f%%".format(100 * this.validate(testSet)))
    }
  }

  /**
   *
   */
  private fun loadExamples(): ArrayList<Example> {

    val examples = arrayListOf<Example>()

    File(this.trainingSetPath).forEachLine { line ->

      val data: List<Double> = line.split(",").map { it.toDouble() }

      val input1 = DenseNDArrayFactory.arrayOf(data.subList(0, 5).toDoubleArray())
      val input2 = DenseNDArrayFactory.arrayOf(data.subList(5, 10).toDoubleArray())
      val output = DenseNDArrayFactory.arrayOf(data.subList(10, 15).toDoubleArray())

      examples.add(Triple(input1, input2, output))
    }

    return examples
  }

  /**
   *
   */
  private fun trainEpoch(trainingExamples: List<Example>) {

    this.loopExamples(trainingExamples) { example ->

      this.optimizer.newEpoch()

      this.trainExample(example)
    }
  }

  /**
   *
   */
  private fun validate(testExamples: List<Example>): Double {

    var correctPredictions = 0

    this.loopExamples(testExamples) { example ->

      val output: DenseNDArray = this.predict(example)

      if (example.third.equals(output, tolerance = 0.01)) {
        correctPredictions++
      }
    }

    return correctPredictions.toDouble() / testExamples.size
  }

  /**
   *
   */
  private fun trainExample(example: Example) {

    this.optimizer.newBatch()
    this.optimizer.newExample()

    this.biaffineProcessor.backward(outputErrors = this.predict(example).sub(example.third))

    this.optimizer.accumulate(this.biaffineProcessor.getParamsErrors(copy = false))
    this.optimizer.update()
  }

  /**
   *
   */
  private fun predict(example: Example): DenseNDArray {

    this.biaffineProcessor.forward(listOf(example.first, example.second))

    return this.biaffineProcessor.getOutput()
  }

  /**
   *
   */
  private fun loopExamples(examples: List<Example>, callback: (example: Example) -> Unit) {

    for (exampleIndex in ShuffledIndices(examples.size, shuffler = this.shuffler)) {
      callback(examples[exampleIndex])
    }
  }
}
