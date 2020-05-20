/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package mnist

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.FeedforwardNeuralNetwork
import traininghelpers.training.FeedforwardTrainer
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.ClassificationEvaluation
import traininghelpers.validation.FeedforwardEvaluator
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import utils.exampleextractor.ClassificationExampleExtractor
import utils.CorpusReader
import Configuration
import com.kotlinnlp.simplednn.core.functionalities.activations.ReLU
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam.RADAMMethod
import utils.Corpus
import utils.SimpleExample

fun main() {

  println("Start 'MNIST Test'")

  val dataset = CorpusReader<SimpleExample<DenseNDArray>>().read(
    corpusPath = Configuration.loadFromFile().mnist.datasets_paths,
    exampleExtractor = ClassificationExampleExtractor(outputSize = 10),
    perLine = false)

  MNISTTest(dataset).start()

  println("\nEnd.")
}

/**
 *
 */
private class MNISTTest(val dataset: Corpus<SimpleExample<DenseNDArray>>) {

  /**
   *
   */
  private val neuralNetwork = FeedforwardNeuralNetwork(
    inputSize = 784,
    hiddenSize = 500,
    hiddenActivation = ReLU,
    outputSize = 10,
    outputActivation = Softmax())

  /**
   *
   */
  fun start() {

    println("\n-- TRAINING")

    FeedforwardTrainer(
      model = this.neuralNetwork,
      updateMethod = RADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999),
      lossCalculator = SoftmaxCrossEntropyCalculator,
      examples = this.dataset.training,
      epochs = 15,
      batchSize = 1,
      evaluator = FeedforwardEvaluator(
        model = this.neuralNetwork,
        examples = this.dataset.validation,
        outputEvaluationFunction = ClassificationEvaluation)
    ).train()
  }
}
