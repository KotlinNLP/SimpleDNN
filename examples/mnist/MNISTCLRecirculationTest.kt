/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package mnist

import com.kotlinnlp.simplednn.dataset.*
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import utils.CorpusReader
import Configuration
import com.kotlinnlp.simplednn.deeplearning.competitivelearning.recirculation.CLRecirculationNetwork
import com.kotlinnlp.simplednn.deeplearning.competitivelearning.recirculation.CLRecirculationModel
import com.kotlinnlp.simplednn.helpers.training.CLRecirculationTrainingHelper
import com.kotlinnlp.simplednn.helpers.validation.CLValidationHelper
import utils.exampleextractor.ClassificationBinaryOutputExampleExtractor

fun main(args: Array<String>) {

  println("Start 'MNIST CL Recirculation Test'")

  val dataset = CorpusReader<BinaryOutputExample<DenseNDArray>>().read(
    corpusPath = Configuration.loadFromFile().mnist.datasets_paths,
    exampleExtractor = ClassificationBinaryOutputExampleExtractor(),
    perLine = false)

  MNISTCLRecirculationTest(dataset).start()

  println("End.")
}

/**
 *
 */
class MNISTCLRecirculationTest(val dataset: Corpus<BinaryOutputExample<DenseNDArray>>) {

  /**
   *
   */
  private val model = CLRecirculationModel(
    numOfClasses = 10,
    inputSize = 784,
    hiddenSize = 50)

  /**
   *
   */
  fun start() {
    this.train()
  }

  /**
   *
   */
  private fun train() {

    println("\n-- TRAINING")

    val trainingHelper = CLRecirculationTrainingHelper(network = CLRecirculationNetwork(this.model), verbose = true)
    val validationHelper = CLValidationHelper(network = CLRecirculationNetwork(this.model))

    trainingHelper.train(
      trainingExamples = this.dataset.training,
      validationExamples = this.dataset.validation,
      epochs = 15,
      batchSize = 1,
      shuffler = Shuffler(enablePseudoRandom = true, seed = 1),
      validationHelper = validationHelper)
  }
}
