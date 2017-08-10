/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork

import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.utils.ItemsPool

/**
 * A pool of [AttentionNetwork]s which allows to allocate and release one when needed, without creating a new one.
 * It is useful to optimize the creation of new structures every time a new encoder is created.
 *
 * @property model the model of the [AttentionNetwork]s of the pool
 * @property inputType the type of the input arrays
 * @property dropout the probability of dropout (default 0.0) when generating the attention arrays for the Attention
 *                   Layer. If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 */
class AttentionNetworksPool<InputNDArrayType : NDArray<InputNDArrayType>>(
  val model: AttentionNetworkParameters,
  val inputType: LayerType.Input,
  val dropout: Double = 0.0
) : ItemsPool<AttentionNetwork<InputNDArrayType>>() {

  /**
   * The factory of a new [AttentionNetwork].
   *
   * @param id the unique id of the item to create
   *
   * @return a new [AttentionNetwork] with the given [id]
   */
  override fun itemFactory(id: Int) = AttentionNetwork<InputNDArrayType>(
    model = this.model,
    inputType = this.inputType,
    dropout = this.dropout, id = id
  )
}
