/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.types.merge.product

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import com.kotlinnlp.utils.ItemsPool

/**
 * A pool of [ProductLayerStructure]s which allows to allocate and release layers when needed, without creating
 * a new one every time.
 *
 * @property params the parameters which connect the input to the output
 * @property inputType the type of the input array
 */
class ProductLayersPool<InputNDArrayType : NDArray<InputNDArrayType>>(
  val params: ProductLayerParameters,
  val inputType: LayerType.Input
) : ItemsPool<ProductLayerStructure<InputNDArrayType>>() {

  /**
   * The factory of a new layer structure.
   *
   * @param id the id of the processor to create
   *
   * @return a new [ProductLayerStructure] with the given [id]
   */
  override fun itemFactory(id: Int): ProductLayerStructure<InputNDArrayType> {

    @Suppress("UNCHECKED_CAST")
    val inputArrays: List<AugmentedArray<InputNDArrayType>> = List(
      size = this.params.nInputs,
      init = {
        when (this.inputType) {
          LayerType.Input.Dense -> AugmentedArray<DenseNDArray>(size = this.params.inputSize)
          LayerType.Input.Sparse -> AugmentedArray<SparseNDArray>(size = this.params.inputSize)
          LayerType.Input.SparseBinary -> AugmentedArray<SparseBinaryNDArray>(size = this.params.inputSize)
        } as AugmentedArray<InputNDArrayType>
      }
    )

    return ProductLayerStructure(
      inputArrays = inputArrays,
      outputArray = AugmentedArray(this.params.outputSize),
      params = this.params,
      id = id
    )
  }
}
