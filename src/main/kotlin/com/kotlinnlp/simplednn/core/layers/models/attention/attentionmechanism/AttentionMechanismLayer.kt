/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.attention.attentionmechanism

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.SoftmaxBase
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The Attention Mechanism Layer Structure.
 *
 * @property inputArrays the input arrays of the layer
 * @param inputType the input array type (default Dense)
 * @param params the parameters which connect the input to the output
 * @param activation the activation function of the layer (default SoftmaxBase)
 */
internal class AttentionMechanismLayer(
  val inputArrays: List<AugmentedArray<DenseNDArray>>,
  inputType: LayerType.Input,
  override val params: AttentionMechanismLayerParameters,
  activation: ActivationFunction? = SoftmaxBase()
) : Layer<DenseNDArray>(
  inputArray = AugmentedArray(params.inputSize), // empty array (it should not be used)
  inputType = inputType,
  outputArray = AugmentedArray(inputArrays.size),
  params = params,
  activationFunction = activation,
  dropout = 0.0
) {

  /**
   * A matrix containing the attention arrays as rows.
   */
  internal val attentionMatrix: AugmentedArray<DenseNDArray> = AugmentedArray(
    DenseNDArrayFactory.fromRows(this.inputArrays.map { it.values })
  )

  /**
   * The helper which executes the forward.
   */
  override val forwardHelper = AttentionMechanismForwardHelper(layer = this)

  /**
   * The helper which executes the backward.
   */
  override val backwardHelper = AttentionMechanismBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance.
   */
  override val relevanceHelper: RelevanceHelper? = null

  /**
   * Initialization: set the activation function of the output array.
   */
  init {

    require(this.inputArrays.isNotEmpty()) { "The attention sequence cannot be empty." }
    require(this.inputArrays.all { it.values.length == this.params.inputSize }) {
      "The input arrays must have the expected size (${this.params.inputSize})."
    }

    if (activation != null)
      this.outputArray.setActivation(activation)
  }
}
