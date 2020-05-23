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
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArrayMask
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import java.lang.Math.random

/**
 * The Scaled-Dot Attention Layer structure.
 *
 * @property inputArrays the input arrays of the layer
 * @property params the parameters which connect the input to the output
 * @property dropout the probability of attention dropout (default 0.0)
 * @param attentionDropout the probability of attention dropout (default 0.0)
 */
internal class ScaledDotAttentionLayer(
  val inputArrays: List<AugmentedArray<DenseNDArray>>,
  override val params: ScaledDotAttentionLayerParameters,
  private val attentionDropout: Double = 0.0
) : Layer<DenseNDArray>(
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
   * The dropout masks of the activated attention column arrays.
   */
  internal var dropoutMasks: List<NDArrayMask>? = if (this.attentionDropout > 0.0) this.buildDropoutMasks() else null

  /**
   * The masks of the attention arrays concatenated horizontally in a unique matrix.
   */
  internal var dropoutMaskFull: NDArrayMask? = this.dropoutMasks?.let { masks ->
    NDArrayMask(
      dim1 = masks.fold(intArrayOf()) { dim1Indices, mask -> dim1Indices + mask.dim1},
      dim2 = masks.foldIndexed(intArrayOf()) { j, dim2Indices, mask ->
        dim2Indices + IntArray(size = mask.size, init = { j })
      })
  }

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

  /**
   * @return the dropout masks of the attention column arrays
   */
  private fun buildDropoutMasks(): List<NDArrayMask> = List(
    size = this.inputArrays.size,
    init = {

      val activeIndices: List<Int> = this.inputArrays.indices
        .asSequence()
        .map { it to random() }
        .filter { it.second >= this.attentionDropout }
        .map { it.first }
        .toList()

      val dim2Indices = IntArray(size = activeIndices.size, init = { 0 })

      NDArrayMask(dim1 = activeIndices.toIntArray(), dim2 = dim2Indices)
    }
  )
}
