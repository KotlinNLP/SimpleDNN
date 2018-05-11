/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.attention.han

import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.utils.ItemsPool

/**
 * A pool of [HANEncoder]s which allows to allocate and release one when needed, without creating a new one.
 * It is useful to optimize the creation of new structures every time a new encoder is created.
 *
 * @property model the HAN model of the [HANEncoder]s of the pool
 */
class HANEncodersPool<InputNDArrayType : NDArray<InputNDArrayType>>(
  val model: HAN
) : ItemsPool<HANEncoder<InputNDArrayType>>() {

  /**
   * The factory of a new [HANEncoder].
   *
   * @param id the unique id of the item to create
   *
   * @return a new [HANEncoder] with the given [id]
   */
  override fun itemFactory(id: Int) = HANEncoder<InputNDArrayType>(model = this.model, id = id)
}
