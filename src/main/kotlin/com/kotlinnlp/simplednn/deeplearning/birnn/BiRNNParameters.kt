/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn

import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import java.io.Serializable

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
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }
}
