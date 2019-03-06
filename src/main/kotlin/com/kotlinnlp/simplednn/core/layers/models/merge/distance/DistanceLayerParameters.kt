/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.distance

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.layers.models.merge.MergeLayerParameters

/**
 * The parameters of the Distance layer.
 *
 * @property inputSize the size of each input
 */
class DistanceLayerParameters(
    inputSize: Int
) : MergeLayerParameters<DistanceLayerParameters>(
    inputsSize = List(size = 2, init = { inputSize }),
    outputSize = 1,
    weightsInitializer = null,
    biasesInitializer = null,
    sparseInput = false // actually not used because there are no parameters
){

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The list of all parameters.
   */
  override val paramsList = emptyList<UpdatableArray<*>>()

  /**
   * The list of weights parameters.
   */
  override val weightsList = emptyList<UpdatableArray<*>>()

  /**
   * The list of biases parameters.
   */
  override val biasesList = emptyList<UpdatableArray<*>>()

  /**
   * @return a new [DistanceLayerParameters] containing a copy of all parameters of this
   */
  override fun copy() = DistanceLayerParameters(inputSize = this.inputSize)
}