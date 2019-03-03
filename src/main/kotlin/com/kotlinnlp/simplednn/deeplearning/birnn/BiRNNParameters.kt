/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.optimizer.IterableParams

/**
 * The BiRNNParameters contains the parameter of its sub-networks (leftToRightNetwork, rightToLeftNetwork).
 *
 * @property leftToRight network parameters of the left-to-right recurrent neural network
 * @property rightToLeft network parameters of the right-to-left recurrent neural network
 * @property merge network parameters of the merge output network
 */
class BiRNNParameters(
  val leftToRight: StackedLayersParameters,
  val rightToLeft: StackedLayersParameters,
  val merge: StackedLayersParameters
) : IterableParams<BiRNNParameters>() {

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
  override val paramsList: List<UpdatableArray<*>> =
    this.leftToRight.paramsList + this.rightToLeft.paramsList + this.merge.paramsList

  /**
   * @return a new [BiRNNParameters] containing a copy of all values of this
   */
  override fun copy() = BiRNNParameters(
    leftToRight = this.leftToRight.copy(),
    rightToLeft = this.rightToLeft.copy(),
    merge = this.merge.copy()
  )
}
