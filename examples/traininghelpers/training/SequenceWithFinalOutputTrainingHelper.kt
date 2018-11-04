/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package traininghelpers.training

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.functionalities.losses.LossCalculator
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import utils.SequenceExampleWithFinalOutput

/**
 * @property optimizer the optimizer
 * @property verbose whether to print training details
 */
class SequenceWithFinalOutputTrainingHelper<NDArrayType: NDArray<NDArrayType>>(
  val neuralProcessor: RecurrentNeuralProcessor<NDArrayType>,
  override val optimizer: ParamsOptimizer<NetworkParameters>,
  val lossCalculator: LossCalculator,
  verbose: Boolean = false
) : TrainingHelper<SequenceExampleWithFinalOutput<NDArrayType>>(
  optimizer = optimizer,
  verbose = verbose) {

  /**
   * Require softmax cross-entropy loss to be used with the softmax as output activation function and vice versa.
   */
  init {

    val activation = this.neuralProcessor.neuralNetwork.layersConfiguration.last().activationFunction

    require(
      (this.lossCalculator is SoftmaxCrossEntropyCalculator && activation is Softmax) ||
        (this.lossCalculator !is SoftmaxCrossEntropyCalculator && activation !is Softmax)
    ) {
      "Softmax cross-entropy loss must be used with the softmax as output activation function and vice versa"
    }
  }

  /**
   * Learn from an example (forward + backward)
   *
   * @param example the example used to train the network
   *
   * @return the loss of the output respect to the gold
   */
  override fun learnFromExample(example: SequenceExampleWithFinalOutput<NDArrayType>): Double {

    this.neuralProcessor.forward(example.sequenceFeatures)

    val output = this.neuralProcessor.getOutput(copy = true)
    val outputError = this.lossCalculator.calculateErrors(output = output, outputGold = example.outputGold)

    this.neuralProcessor.backward(outputError)

    return this.lossCalculator.calculateLoss(output = output, outputGold = example.outputGold).avg()
  }

  /**
   * Accumulate the params errors resulting from [learnFromExample].
   *
   * @param batchSize the size of each batch
   */
  override fun accumulateParamsErrors(batchSize: Int) {
    this.optimizer.accumulate(this.neuralProcessor.getParamsErrors(copy = batchSize > 1), copy = batchSize > 1)
  }
}
