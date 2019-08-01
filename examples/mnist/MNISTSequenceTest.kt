/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package mnist

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.neuralnetwork.preset.GRUNeuralNetwork
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.functionalities.outputevaluation.ClassificationEvaluation
import traininghelpers.training.SequenceWithFinalOutputTrainingHelper
import traininghelpers.validation.SequenceWithFinalOutputValidationHelper
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.Shuffler
import utils.Corpus
import utils.SequenceExampleWithFinalOutput
import mnist.helpers.MNISTSequenceExampleExtractor
import utils.CorpusReader

fun main() {

  println("Start 'MNIST Sequence Test'")

  val dataset = CorpusReader<SequenceExampleWithFinalOutput<DenseNDArray>>().read(
    corpusPath = Configuration.loadFromFile().mnist_sequence.datasets_paths, // same for validation and test
    exampleExtractor = MNISTSequenceExampleExtractor(outputSize = 10),
    perLine = true)

  MNISTSequenceTest(dataset).start()

  println("\nEnd.")
}

/**
 *
 */
class MNISTSequenceTest(val dataset: Corpus<SequenceExampleWithFinalOutput<DenseNDArray>>) {

  /**
   *
   */
  val neuralNetwork = GRUNeuralNetwork(
    inputSize = 2,
    hiddenSize = 200,
    hiddenActivation = Tanh(),
    outputSize = 10,
    outputActivation = Softmax())

  /**
   *
   */
  fun start() {

    this.train()
  }

  /**
   *
   */
  fun train() {

    println("\n-- TRAINING")

    val optimizer = ParamsOptimizer(updateMethod = ADAMMethod(stepSize = 0.001))

    val neuralProcessor = RecurrentNeuralProcessor<DenseNDArray>(
      model = this.neuralNetwork,
      useDropout = false,
      propagateToInput = false)

    val trainingHelper = SequenceWithFinalOutputTrainingHelper(
      neuralProcessor = neuralProcessor,
      optimizer = optimizer,
      lossCalculator = SoftmaxCrossEntropyCalculator(),
      verbose = true)

    val validationHelper = SequenceWithFinalOutputValidationHelper(
      neuralProcessor = neuralProcessor,
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
