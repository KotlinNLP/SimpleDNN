/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn

import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.utils.ItemsPool

/**
 * A pool of [BiRNNEncoder]s which allows to allocate and release one when needed, without creating a new one.
 * It is useful to optimize the creation of new structures every time a new encoder is created.
 *
 * @param network the [BiRNN] which the encoders of the pool will work with
 * @param rnnDropout the probability of RNNs dropout (default 0.0)
 * @param mergeDropout the probability of output merge dropout (default 0.0)
 * @param propagateToInput whether to propagate the errors to the input during the backward
 */
class BiRNNEncodersPool<InputNDArrayType : NDArray<InputNDArrayType>>(
  private val network: BiRNN,
  private val rnnDropout: Double = 0.0,
  private val mergeDropout: Double = 0.0,
  private val propagateToInput: Boolean
) : ItemsPool<BiRNNEncoder<InputNDArrayType>>() {

  /**
   * The factory of a new [BiRNNEncoder].
   *
   * @param id the unique id of the item to create
   *
   * @return a new [BiRNNEncoder] with the given [id]
   */
  override fun itemFactory(id: Int) = BiRNNEncoder<InputNDArrayType>(
    network = this.network,
    rnnDropout = this.rnnDropout,
    mergeDropout = this.mergeDropout,
    propagateToInput = this.propagateToInput,
    id = id
  )
}
