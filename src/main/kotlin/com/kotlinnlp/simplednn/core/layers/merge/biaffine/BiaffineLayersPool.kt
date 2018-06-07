/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.merge.biaffine

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.LayerUnit
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import com.kotlinnlp.utils.ItemsPool

/**
 * A pool of [BiaffineLayerStructure]s which allows to allocate and release layers when needed, without creating
 * a new one every time.
 *
 * @property params the parameters which connect the input to the output
 * @property inputType the type of the input array
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default: 0.0 - if applying it, the usual value is 0.25)
 */
class BiaffineLayersPool<InputNDArrayType : NDArray<InputNDArrayType>>(
  val params: BiaffineLayerParameters,
  val inputType: LayerType.Input,
  val activationFunction: ActivationFunction?,
  val dropout: Double = 0.0
) : ItemsPool<BiaffineLayerStructure<InputNDArrayType>>() {

  /**
   * The factory of a new layer structure.
   *
   * @param id the id of the processor to create
   *
   * @return a new [BiaffineLayerStructure] with the given [id]
   */
  override fun itemFactory(id: Int): BiaffineLayerStructure<InputNDArrayType> {

    val (inputArray1, inputArray2) = when (this.inputType) {

      LayerType.Input.Dense -> Pair(
        AugmentedArray<DenseNDArray>(size = this.params.inputSize),
        AugmentedArray<DenseNDArray>(size = this.params.inputSize))

      LayerType.Input.Sparse -> Pair(
        AugmentedArray<SparseNDArray>(size = this.params.inputSize),
        AugmentedArray<SparseNDArray>(size = this.params.inputSize))

      LayerType.Input.SparseBinary -> Pair(
        AugmentedArray<SparseBinaryNDArray>(size = this.params.inputSize),
        AugmentedArray<SparseBinaryNDArray>(size = this.params.inputSize))
    }

    @Suppress("UNCHECKED_CAST")
    return BiaffineLayerStructure(
      inputArray1 = inputArray1 as AugmentedArray<InputNDArrayType>,
      inputArray2 = inputArray2 as AugmentedArray<InputNDArrayType>,
      outputArray = LayerUnit<DenseNDArray>(this.params.outputSize),
      params = this.params,
      activationFunction = this.activationFunction,
      dropout = this.dropout,
      id = id
    )
  }
}
