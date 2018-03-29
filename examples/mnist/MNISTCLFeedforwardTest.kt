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
import com.kotlinnlp.simplednn.core.functionalities.activations.Sigmoid
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.deeplearning.competitivelearning.feedforward.CLFeedforwardNetwork
import com.kotlinnlp.simplednn.deeplearning.competitivelearning.feedforward.CLFeedforwardNetworkModel
import com.kotlinnlp.simplednn.deeplearning.competitivelearning.feedforward.CLFeedforwardNetworkOptimizer
import com.kotlinnlp.simplednn.helpers.training.CLFeedforwardTrainingHelper
import com.kotlinnlp.simplednn.helpers.validation.CLValidationHelper
import utils.exampleextractor.ClassificationBinaryOutputExampleExtractor

fun main(args: Array<String>) {

  println("Start 'MNIST CL Feedforward Test'")

  val dataset = CorpusReader<BinaryOutputExample<DenseNDArray>>().read(
    corpusPath = Configuration.loadFromFile().mnist.datasets_paths,
    exampleExtractor = ClassificationBinaryOutputExampleExtractor(),
    perLine = false)

  MNISTCLFeedforwardTest(dataset).start()

  println("End.")
}

/**
 *
 */
class MNISTCLFeedforwardTest(val dataset: Corpus<BinaryOutputExample<DenseNDArray>>) {

  /**
   *
   */
  private val model = CLFeedforwardNetworkModel(
    numOfClasses = 10,
    inputSize = 784,
    hiddenSize = 50,
    hiddenActivation = Sigmoid())

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

    val optimizer = CLFeedforwardNetworkOptimizer(
      model = this.model,
      updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999))

    val trainingHelper = CLFeedforwardTrainingHelper(
      network = CLFeedforwardNetwork(this.model),
      optimizer = optimizer,
      verbose = true)

    val validationHelper = CLValidationHelper(network = CLFeedforwardNetwork(this.model))

    trainingHelper.train(
      trainingExamples = this.dataset.training,
      validationExamples = this.dataset.validation,
      epochs = 15,
      batchSize = 1,
      shuffler = Shuffler(enablePseudoRandom = true, seed = 1),
      validationHelper = validationHelper)
  }
}
