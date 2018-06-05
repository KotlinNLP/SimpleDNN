/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.mergelayers.product

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerUnit
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.ItemsPool

/**
 * A pool of [ProductLayerStructure]s which allows to allocate and release layers when needed, without creating
 * a new one every time.
 *
 * @property params the parameters which connect the input to the output
 */
class ProductLayersPool(val params: ProductLayerParameters) : ItemsPool<ProductLayerStructure>() {

  /**
   * The factory of a new layer structure.
   *
   * @param id the id of the processor to create
   *
   * @return a new [ProductLayerStructure] with the given [id]
   */
  override fun itemFactory(id: Int): ProductLayerStructure {

    val inputArrays: List<AugmentedArray<DenseNDArray>> =
      List(size = this.params.inputsSize.size, init = {
        AugmentedArray<DenseNDArray>(size = this.params.inputsSize[it])
      })

    return ProductLayerStructure(
      inputArrays = inputArrays,
      outputArray = LayerUnit<DenseNDArray>(this.params.outputSize),
      params = this.params,
      id = id
    )
  }
}
