/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attention.han

import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.utils.ItemsPool

/**
 * A pool of [HANEncoder]s which allows to allocate and release one when needed, without creating a new one.
 * It is useful to optimize the creation of new structures every time a new encoder is created.
 *
 * @property model the HAN model of the [HANEncoder]s of the pool
 * @param useDropout whether to apply the dropout during the forward
 * @param propagateToInput whether to propagate the errors to the input during the backward
 * @param mePropK the k factor of the 'meProp' algorithm to propagate from the k (in percentage) output nodes with
 *                the top errors of the transform layers (ignored if null, the default)
 */
class HANEncodersPool<InputNDArrayType : NDArray<InputNDArrayType>>(
  val model: HAN,
  private val useDropout: Boolean,
  private val propagateToInput: Boolean,
  private val mePropK: Double? = null
) : ItemsPool<HANEncoder<InputNDArrayType>>() {

  /**
   * The factory of a new [HANEncoder].
   *
   * @param id the unique id of the item to create
   *
   * @return a new [HANEncoder] with the given [id]
   */
  override fun itemFactory(id: Int) = HANEncoder<InputNDArrayType>(
    model = this.model,
    useDropout = this.useDropout,
    propagateToInput = this.propagateToInput,
    mePropK = this.mePropK,
    id = id)
}
