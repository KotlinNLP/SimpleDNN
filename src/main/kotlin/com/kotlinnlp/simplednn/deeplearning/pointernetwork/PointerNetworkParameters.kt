/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.pointernetwork

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.IterableParams
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionmechanism.AttentionParameters

/**
 * The parameters of the [PointerNetwork].
 *
 * @property recurrentParams the parameters of the recurrent network
 */
class PointerNetworkParameters(
  val recurrentParams: NetworkParameters,
  val transformParams: NetworkParameters,
  val attentionParams: AttentionParameters
) : IterableParams<PointerNetworkParameters>() {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }
  /**
   * The list of all parameters.
   */
  override val paramsList: Array<UpdatableArray<*>> =
      this.recurrentParams.paramsList +
        this.transformParams.paramsList +
        this.attentionParams.paramsList

  /**
   * @return a new [PointerNetworkParameters] containing a copy of all values of this
   */
  override fun copy() = PointerNetworkParameters(
    recurrentParams = this.recurrentParams.copy(),
    transformParams = this.transformParams.copy(),
    attentionParams = this.attentionParams.copy())
}
