/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.competitivelearning.recirculation

import com.kotlinnlp.simplednn.deeplearning.competitivelearning.CLNetwork
import com.kotlinnlp.simplednn.deeplearning.newrecirculation.NewRecirculationNetwork
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The Competitive Learning network based on the New Recirculation.
 *
 * @property model the model of this CL network
 * @param trainingLearningRate the learning rate of the training (default = 0.01)
 */
class CLRecirculationNetwork(
  val model: CLRecirculationModel,
  private val trainingLearningRate: Double = 0.01
): CLNetwork(model) {

  /**
   * The list of New Recirculation networks, one per class.
   */
  private val autoencoders: List<NewRecirculationNetwork> = this.model.classes.map {
    NewRecirculationNetwork(model = this.model.autoencoders[it], trainingLearningRate = this.trainingLearningRate)
  }

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

    return this.autoencoders[classIndex].learn(inputArray)
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

    val encoder: NewRecirculationNetwork = this.autoencoders[classIndex]

    encoder.reconstruct(inputArray, useReEntry = false)

    return encoder.meanAbsError
  }
}
