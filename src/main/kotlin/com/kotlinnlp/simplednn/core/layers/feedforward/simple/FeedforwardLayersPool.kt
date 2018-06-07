/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.feedforward.simple

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.LayerUnit
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import com.kotlinnlp.utils.ItemsPool

/**
 * A pool of [FeedforwardLayerStructure]s which allows to allocate and release layers when needed, without creating
 * a new one every time.
 *
 * @property params the parameters which connect the input to the output
 * @property inputType the type of the input array
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout (default: 0.0 - if applying it, the usual value is 0.25)
 */
class FeedforwardLayersPool<InputNDArrayType : NDArray<InputNDArrayType>>(
  val params: LayerParameters<*>,
  val inputType: LayerType.Input,
  val activationFunction: ActivationFunction?,
  val dropout: Double = 0.0
) : ItemsPool<FeedforwardLayerStructure<InputNDArrayType>>() {

  /**
   * The factory of a new layer structure.
   *
   * @param id the id of the processor to create
   *
   * @return a new [FeedforwardLayerStructure] with the given [id]
   */
  override fun itemFactory(id: Int): FeedforwardLayerStructure<InputNDArrayType> {

    @Suppress("UNCHECKED_CAST")
    val inputArray = when (this.inputType) {
      LayerType.Input.Dense -> AugmentedArray<DenseNDArray>(size = this.params.inputSize)
      LayerType.Input.Sparse -> AugmentedArray<SparseNDArray>(size = this.params.inputSize)
      LayerType.Input.SparseBinary -> AugmentedArray<SparseBinaryNDArray>(size = this.params.inputSize)
    } as AugmentedArray<InputNDArrayType>

    return FeedforwardLayerStructure(
      inputArray = inputArray,
      outputArray = LayerUnit(this.params.outputSize),
      params = this.params,
      activationFunction = this.activationFunction,
      dropout = this.dropout,
      id = id
    )
  }
}
