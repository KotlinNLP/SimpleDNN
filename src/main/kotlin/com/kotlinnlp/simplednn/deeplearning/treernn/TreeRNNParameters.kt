/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.treernn

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.IterableParams

/**
 * The TreeRNNParameters contains the parameter of its sub-networks
 * (leftRNN, rightRNN, concatNetwork)
 *
 * @property leftRNN network parameters of the left recurrent neural network
 * @property rightRNN network parameters of the right recurrent neural network
 * @property concatNetwork network parameters of the final feed-forward network
 */
class TreeRNNParameters(
  val leftRNN: NetworkParameters,
  val rightRNN: NetworkParameters,
  val concatNetwork: NetworkParameters
) : IterableParams<TreeRNNParameters>() {

  /**
   * The list of all parameters.
   */
  override val paramsList: Array<UpdatableArray<*>> = this.buildParamsList()

  /**
   * @return a new [TreeRNNParameters] containing a copy of all parameters of this
   */
  override fun copy() = TreeRNNParameters(
    leftRNN = this.leftRNN.copy(),
    rightRNN = this.rightRNN.copy(),
    concatNetwork= this.concatNetwork.copy()
  )

  /**
   * @return the list with parameters of all layers
   */
  private fun buildParamsList(): Array<UpdatableArray<*>>
    = this.leftRNN.paramsList + this.rightRNN.paramsList + this.concatNetwork.paramsList
}
