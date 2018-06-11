/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.concat

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import com.kotlinnlp.utils.ItemsPool

/**
 * A pool of [ConcatLayerStructure]s which allows to allocate and release layers when needed, without creating
 * a new one every time.
 *
 * @property params the parameters which connect the input to the output
 * @property inputType the type of the input array
 */
class ConcatLayersPool<InputNDArrayType : NDArray<InputNDArrayType>>(
  val params: ConcatLayerParameters,
  val inputType: LayerType.Input
) : ItemsPool<ConcatLayerStructure<InputNDArrayType>>() {

  /**
   * The factory of a new layer structure.
   *
   * @param id the id of the processor to create
   *
   * @return a new [ConcatLayerStructure] with the given [id]
   */
  override fun itemFactory(id: Int): ConcatLayerStructure<InputNDArrayType> {

    @Suppress("UNCHECKED_CAST")
    val inputArrays: List<AugmentedArray<InputNDArrayType>> = this.params.inputsSize.map {
      when (this.inputType) {
        LayerType.Input.Dense -> AugmentedArray<DenseNDArray>(size = it)
        LayerType.Input.Sparse -> AugmentedArray<SparseNDArray>(size = it)
        LayerType.Input.SparseBinary -> AugmentedArray<SparseBinaryNDArray>(size = it)
      } as AugmentedArray<InputNDArrayType>
    }

    return ConcatLayerStructure(
      inputArrays = inputArrays,
      outputArray = AugmentedArray(this.params.outputSize),
      params = this.params,
      id = id
    )
  }
}
