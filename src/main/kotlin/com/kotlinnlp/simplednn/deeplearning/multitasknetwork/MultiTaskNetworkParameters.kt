/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.multitasknetwork

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.IterableParams
import java.io.Serializable

/**
 * The neural parameters of the [MultiTaskNetwork].
 */
class MultiTaskNetworkParameters(
  val inputParams: NetworkParameters,
  val outputParamsList: List<NetworkParameters>
) : Serializable, IterableParams<MultiTaskNetworkParameters>() {

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
  override val paramsList: List<UpdatableArray<*>> = this.outputParamsList.flatMap { it.paramsList }

  /**
   * @return a new [MultiTaskNetworkParameters] containing a copy of all parameters of this
   */
  override fun copy() = MultiTaskNetworkParameters(
    this.inputParams.copy(),
    this.outputParamsList.map { it.copy() }
  )
}

