/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.types.merge.biaffine

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.ItemsPool

/**
 * A pool of [BiaffineLayerStructure]s with dense input, which allows to allocate and release one when needed, without
 * creating a new one.
 * It is useful to optimize the creation of new structures every time a new encoder is created.
 *
 * @property model the model of a dense biaffine layer
 * @property dropout the probability of dropout (default = 0.0 -> if applying it, the usual value is 0.25)
 */
class DenseBiaffineLayersPool(
  val model: BiaffineLayerModel,
  val dropout: Double = 0.0
) : ItemsPool<BiaffineLayerStructure<DenseNDArray>>() {

  /**
   * The factory of a new [BiaffineLayerStructure].
   *
   * @param id the unique id of the item to create
   *
   * @return a new [BiaffineLayerStructure] with the given [id]
   */
  override fun itemFactory(id: Int) = BiaffineLayerStructure<DenseNDArray>(
    params = this.model.params,
    activationFunction = this.model.activationFunction,
    dropout = this.dropout,
    id = id)
}
