/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.mergelayers.sum

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerUnit
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.utils.ItemsPool

/**
 * A pool of [SumLayerStructure]s which allows to allocate and release layers when needed, without creating
 * a new one every time.
 *
 * @property params the parameters which connect the input to the output
 */
class SumLayersPool(val params: SumLayerParameters) : ItemsPool<SumLayerStructure>() {

  /**
   * The factory of a new layer structure.
   *
   * @param id the id of the processor to create
   *
   * @return a new [SumLayerStructure] with the given [id]
   */
  override fun itemFactory(id: Int): SumLayerStructure {

    val inputArrays: List<AugmentedArray<DenseNDArray>> =
      List(size = this.params.inputsSize.size, init = {
        AugmentedArray<DenseNDArray>(size = this.params.inputsSize[it])
      })

    return SumLayerStructure(
      inputArrays = inputArrays,
      outputArray = LayerUnit<DenseNDArray>(this.params.outputSize),
      params = this.params,
      id = id
    )
  }
}
