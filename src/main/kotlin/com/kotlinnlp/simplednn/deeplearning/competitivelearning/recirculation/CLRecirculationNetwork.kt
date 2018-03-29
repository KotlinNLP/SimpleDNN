/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.competitivelearning.recirculation

import com.kotlinnlp.simplednn.deeplearning.newrecirculation.NewRecirculationNetwork
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The Competitive Learning Network based on the New Recirculation.
 *
 * @property model the model of this network
 * @param trainingLearningRate the learning rate of the training (default = 0.01)
 */
class CLRecirculationNetwork(val model: CLRecirculationModel, private val trainingLearningRate: Double = 0.01) {

  /**
   * The scores of the last prediction, one per encoder.
   */
  val losses: List<Double> get() = this._losses

  /**
   * The scores of the last prediction, one per encoder, updated dynamically.
   */
  private val _losses = mutableListOf<Double>()

  /**
   * The map that associates each class to a feed-forward processor.
   * Each processor keeps the class to which it refers in the 'id' property.
   */
  private val autoencoders: Map<Int, NewRecirculationNetwork> = this.model.classes.associate {
    it to NewRecirculationNetwork(
      model = this.model.autoencodersModels.getValue(it),
      trainingLearningRate = this.trainingLearningRate,
      id = it)
  }

  /**
   * Predict using the MSE as distance measure.
   *
   * @param inputArray the input array
   *
   * @return the highest scoring predicted class
   */
  fun predict(inputArray: DenseNDArray): Int =
    this.autoencoders.keys.minBy { this.reconstructAndGetLoss(inputArray = inputArray, encoderIndex = it) }!!

  /**
   * Perform the learning of an example.
   *
   * @param inputArray the input array
   * @param classIndex the index of the class assigned to the given [inputArray]
   *
   * @return the loss of the reconstruction
   */
  fun learn(inputArray: DenseNDArray, classIndex: Int): Double {

    val autoencoder: NewRecirculationNetwork =
      checkNotNull(this.autoencoders[classIndex]) { "Unknown class: $classIndex" }

    return autoencoder.learn(inputArray)
  }

  /**
   * Reconstruct a given array with a given encoder and get the mean abs error of the reconstruction.
   *
   * @param inputArray the input array
   * @param encoderIndex the encoder index
   *
   * @return the loss of the reconstruction
   */
  private fun reconstructAndGetLoss(inputArray: DenseNDArray, encoderIndex: Int): Double {

    val encoder: NewRecirculationNetwork = this.autoencoders.getValue(encoderIndex)

    encoder.reconstruct(inputArray, useReEntry = false)

    this._losses[encoderIndex] = encoder.meanAbsError

    return encoder.meanAbsError
  }
}
