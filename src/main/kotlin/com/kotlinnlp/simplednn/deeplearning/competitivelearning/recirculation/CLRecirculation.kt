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
 */
class CLRecirculation(val model: CLRecirculationModel) {

  /**
   * The scores of the last prediction.
   */
  val outputScores: Map<Int, Double> get() = this.mutableScores

  /**
   * The mutable map which contains the scores of the last prediction.
   */
  private val mutableScores = mutableMapOf<Int, Double>()

  /**
   * The map that associates each class to a feed-forward processor.
   * Each processor keeps the class to which it refers in the 'id' property.
   */
  private val autoencoders: Map<Int, NewRecirculationNetwork> =
    this.model.classes.associate { it to NewRecirculationNetwork(
      model = this.model.autoencodersModels.getValue(it),
      trainingLearningRate = 0.01,
      id = it) }

  /**
   * Predict using the MSE as distance measure.
   *
   * @param inputArray the input array
   *
   * @return the highest scoring predicted class
   */
  fun predict(inputArray: DenseNDArray): Int {

    return this.autoencoders.minBy { (key, encoder) ->

      encoder.reconstruct(inputArray)
      this.mutableScores[key] = encoder.meanAbsError

      encoder.meanAbsError
    }!!
      .key
  }

  /**
   * Learn.
   *
   * @param inputArray the input array
   * @param classId the class to which the [inputArray] belongs
   *
   * @return
   */
  fun learn(inputArray: DenseNDArray, classId: Int): Double {

    val autoencoder: NewRecirculationNetwork = checkNotNull(this.autoencoders[classId]) {
      "Unknown class: $classId"
    }

    return autoencoder.learn(inputArray)
  }
}