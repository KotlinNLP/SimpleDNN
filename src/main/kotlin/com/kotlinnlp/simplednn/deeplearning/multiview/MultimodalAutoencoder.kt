/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.multiview

import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetwork
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetworkParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * This is an implementation of the multi-view model proposed by Ngiam et al. (2011).
 *
 * This auto-encoder creates a shared representations by reconstructing multiple views available during the learning
 * phase from the one view that is always available at test time.
 *
 * The intuition behind the model is that a shared representation can be extracted from a single view, and can be used
 * to reconstruct all the other views.
 *
 * @param network the multi-task network used as multi-view auto-encoder
 */
class MultimodalAutoencoder(val network: MultiTaskNetwork<DenseNDArray>) {

  /**
   * The Prediction.
   *
   * @property reconstructedViews the reconstructed views
   * @property sharedRepresentation the shared representation
   */
  data class Prediction(
    val reconstructedViews: List<DenseNDArray>,
    val sharedRepresentation: DenseNDArray)

  /**
   * The loss calculator used to calculate the distance between the original and the reconstructed views.
   */
  private val lossCalculator = MSECalculator()

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the neural parameters
   */
  fun getParamsErrors(copy: Boolean): MultiTaskNetworkParameters = this.network.getParamsErrors(copy = copy)

  /**
   * Predict all the views starting from an input array, where the first view is the reconstruction of the input itself.
   *
   * @param inputArray the input array.
   * @param copy a Boolean indicating whether the returned array must be a copy or a reference
   *
   * @return a [Prediction] where the order of the predicted views follows the order in which they were trained
   */
  fun predict(inputArray: DenseNDArray, copy: Boolean = true) = Prediction(
    reconstructedViews = this.network.forward(inputArray).map { if (copy) it.copy() else it },
    sharedRepresentation = this.network.inputProcessor.getOutput(copy = copy))

  /**
   * @param inputArray the input array
   * @param copy a Boolean indicating whether the returned array must be a copy or a reference
   */
  fun getSharedRepresentations(inputArray: DenseNDArray, copy: Boolean = true): DenseNDArray {
    this.network.inputProcessor.forward(featuresArray = inputArray)
    return this.network.inputProcessor.getOutput(copy = copy)
  }

  /**
   * @param views all the views (where the first one is intended as the input array to be reconstructed)
   *
   * @return the loss for this example
   */
  fun learn(views: List<DenseNDArray>): Double {

    require(views.size == this.network.model.outputNetworks.size) {
      "The number of views must be equal to the number of output networks."
    }

    val reconstructedViews: Array<DenseNDArray> = this.network.forward(views.first()).toTypedArray()

    val errors: Array<DenseNDArray> = this.lossCalculator.calculateErrors(
      outputGoldSequence = views.toTypedArray(),
      outputSequence = reconstructedViews)

    this.network.backward(errors.toList())

    return this.lossCalculator.calculateMeanLoss(
      outputGoldSequence = views.toTypedArray(),
      outputSequence = reconstructedViews)
  }
}