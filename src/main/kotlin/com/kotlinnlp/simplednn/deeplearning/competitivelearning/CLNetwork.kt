/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.competitivelearning

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The Competitive Learning network.
 *
 * @param model the model of this CL network
 */
abstract class CLNetwork(private val model: CLNetworkModel) {

  /**
   * The losses of the last prediction, one per encoder.
   */
  val losses: List<Double> get() = this._losses.toList()

  /**
   * The losses of the last prediction, one per encoder, updated dynamically
   */
  private val _losses: Array<Double> = Array(size = this.model.numOfClasses, init = { 0.0 })

  /**
   * Classify an input.
   *
   * @param inputArray the input array
   *
   * @return the highest scoring predicted class index
   */
  fun classify(inputArray: DenseNDArray): Int = this.model.classes.minBy {
    this._losses[it] = this.reconstructAndGetLoss(inputArray = inputArray, classIndex = it)
    this._losses[it]
  }!!

  /**
   * Perform the learning of an example.
   *
   * @param inputArray the input array
   * @param classIndex the index of the class assigned to the given [inputArray]
   *
   * @return the loss of the classification
   */
  abstract fun learn(inputArray: DenseNDArray, classIndex: Int): Double

  /**
   * Reconstruct a given input respect to a given class with the related processor and get the loss of the
   * reconstruction.
   *
   * @param inputArray the input array
   * @param classIndex the class index
   *
   * @return the loss of the reconstruction
   */
  protected abstract fun reconstructAndGetLoss(inputArray: DenseNDArray, classIndex: Int): Double
}
