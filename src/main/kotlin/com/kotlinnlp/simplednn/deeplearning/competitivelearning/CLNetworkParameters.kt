/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.competitivelearning

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.IterableParams

/**
 * The parameters of the [CLNetworkModel].
 *
 * @property networksParams a list of feed-forward network parameters
 */
class CLNetworkParameters(
  private val networksParams: List<NetworkParameters>
) : IterableParams<CLNetworkParameters>() {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The list of all parameters.
   */
  override val paramsList: Array<UpdatableArray<*>> = this.buildParamsList()

  /**
   * @return a new [CLNetworkParameters] containing a copy of all the parameters
   */
  override fun copy() = CLNetworkParameters(this.networksParams.map { it.copy() })

  /**
   * @return the array containing all parameters
   */
  private fun buildParamsList(): Array<UpdatableArray<*>> {

    val params = arrayListOf<UpdatableArray<*>>()

    this.networksParams.forEach { params.addAll(it.paramsList) }

    return params.toTypedArray()
  }
}
