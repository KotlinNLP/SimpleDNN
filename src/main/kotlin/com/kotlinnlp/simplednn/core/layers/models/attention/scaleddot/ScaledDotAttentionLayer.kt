/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.attention.scaleddot

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.utils.ItemsPool

/**
 * The Scaled-Dot Attention Layer structure.
 *
 * @property inputArrays the input arrays of the layer
 * @property params the parameters which connect the input to the output
 * @property id an identification number useful to track a specific layer (default: 0)
 */
class ScaledDotAttentionLayer(
  val inputArrays: List<AugmentedArray<DenseNDArray>>,
  override val params: ScaledDotAttentionLayerParameters,
  override val id: Int = 0
) : ItemsPool.IDItem,
  Layer<DenseNDArray>(
    inputArray = inputArrays[0],
    inputType = LayerType.Input.Dense,
    outputArray = AugmentedArray(inputArrays.size),
    params = params,
    activationFunction = null,
    dropout = 0.0
  ) {

  /**
   * The input matrix with the input arrays as rows.
   */
  val inputMatrix: AugmentedArray<DenseNDArray> =
    AugmentedArray(DenseNDArrayFactory.fromRows(this.inputArrays.map { it.values }))

  /**
   * The output arrays.
   */
  val outputArrays: List<AugmentedArray<DenseNDArray>> = this.inputArrays.map {
    AugmentedArray<DenseNDArray>(this.params.outputSize)
  }

  /**
   * The queries calculated from the input arrays.
   */
  internal val queries: AugmentedArray<DenseNDArray> =
    AugmentedArray(size = this.inputArrays.size * this.params.attentionSize)

  /**
   * The keys calculated from the input arrays.
   */
  internal val keys: AugmentedArray<DenseNDArray> =
    AugmentedArray(size = this.inputArrays.size * this.params.attentionSize)

  /**
   * The values calculated from the input arrays.
   */
  internal val values: AugmentedArray<DenseNDArray> =
    AugmentedArray(size = this.inputArrays.size * this.params.outputSize)

  /**
   * The attention matrix.
   */
  internal lateinit var attention: DenseNDArray

  /**
   * The attention arrays activated with the Softmax function.
   */
  internal lateinit var attentionAct: List<DenseNDArray>

  /**
   * The helper which executes the forward
   */
  override val forwardHelper = ScaledDotAttentionForwardHelper(layer = this)

  /**
   * The helper which executes the backward
   */
  override val backwardHelper = ScaledDotAttentionBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance
   */
  override val relevanceHelper: RelevanceHelper? = null

  /**
   * Check the input size.
   */
  init {
    require(this.inputArrays.isNotEmpty()) { "The attention sequence cannot be empty." }
    require(this.inputArrays.all { it.values.length == this.params.inputSize }) {
      "All the input arrays must have the same size (%d).".format(this.params.inputSize)
    }
  }
}
