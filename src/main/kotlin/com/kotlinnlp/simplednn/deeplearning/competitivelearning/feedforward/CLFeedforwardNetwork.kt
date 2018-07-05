/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.competitivelearning.feedforward

import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.deeplearning.competitivelearning.CLNetwork
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.cosineSimilarity

/**
 * The Competitive Learning network based on the Feedforward.
 *
 * @property model the model of this CL network
 */
class CLFeedforwardNetwork(val model: CLFeedforwardNetworkModel): CLNetwork(model) {

  /**
   * The list of feed-forward processors, one per class.
   */
  private val processors: List<FeedforwardNeuralProcessor<DenseNDArray>> = this.model.classes.map {
    FeedforwardNeuralProcessor<DenseNDArray>(neuralNetwork = this.model.networks[it])
  }

  /**
   * The loss calculator used to calculate the distance between the input and its reconstruction
   */
  private val lossCalculator = MSECalculator()

  /**
   * The processor for which a backward was called the last time.
   */
  private lateinit var lastBackwardProcessor: FeedforwardNeuralProcessor<DenseNDArray>

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the input calculated during the last backward
   */
  fun getInputErrors(copy: Boolean = true): DenseNDArray = this.lastBackwardProcessor.getInputErrors(copy = copy)

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the neural parameters calculated during the last backward
   */
  fun getParamsErrors(copy: Boolean): NetworkParameters = this.lastBackwardProcessor.getParamsErrors(copy = copy)

  /**
   * Perform the learning of an example.
   *
   * @param inputArray the input array
   * @param classIndex the index of the class assigned to the given [inputArray]
   *
   * @return the loss of the classification
   */
  override fun learn(inputArray: DenseNDArray, classIndex: Int): Double {

    require(classIndex in this.model.classes) { "Unknown class: $classIndex." }

    this.lastBackwardProcessor = this.processors[classIndex]

    val reconstructedArray: DenseNDArray = this.lastBackwardProcessor.forward(inputArray)

    this.lastBackwardProcessor.backward(
      outputErrors = this.lossCalculator.calculateErrors(output = reconstructedArray, outputGold = inputArray)
    )

    return this.lossCalculator.calculateLoss(output = reconstructedArray, outputGold = inputArray).avg()
  }

  /**
   * Reconstruct a given input respect to a given class with the related processor and get the loss of the
   * reconstruction.
   *
   * @param inputArray the input array
   * @param classIndex the class index
   *
   * @return the loss of the reconstruction
   */
  override fun reconstructAndGetLoss(inputArray: DenseNDArray, classIndex: Int): Double {

    val reconstructedInput: DenseNDArray = this.processors[classIndex].forward(inputArray)

    return cosineSimilarity(inputArray.normalize2(), reconstructedInput.normalize2())
  }
}
