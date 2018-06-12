/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.sequenceencoder

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.IterableParams

/**
 * The parameters of the [SequenceParallelEncoder].
 *
 * @property networksParams a list of feedforward network parameters
 */
class ParallelEncoderParameters(
  val networksParams: List<NetworkParameters>
) : IterableParams<ParallelEncoderParameters>() {

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
  override val paramsList: List<UpdatableArray<*>> = this.networksParams.flatMap { it.paramsList }

  /**
   * @return a new [ParallelEncoderParameters] containing a copy of all parameters of this
   */
  override fun copy() = ParallelEncoderParameters(this.networksParams.map { it.copy() })
}
