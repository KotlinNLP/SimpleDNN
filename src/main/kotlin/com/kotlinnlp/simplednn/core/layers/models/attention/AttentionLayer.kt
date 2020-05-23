/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.attention

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.SoftmaxBase
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.core.layers.models.attention.attentionmechanism.AttentionMechanismLayer
import com.kotlinnlp.simplednn.core.layers.models.attention.attentionmechanism.AttentionMechanismLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The Attention Layer Structure.
 *
 * @property inputArrays the input arrays of the layer
 * @param inputType the input array type (default Dense)
 * @param params the parameters which connect the input to the output
 * @param activation the activation function of the layer (default SoftmaxBase)
 * @param dropout the probability of dropout (default 0.0).
 *                If applying it, the usual value is 0.5 (better 0.25 if it's the first layer).
 */
internal class AttentionLayer<InputNDArrayType : NDArray<InputNDArrayType>>(
  val inputArrays: List<AugmentedArray<InputNDArrayType>>,
  inputType: LayerType.Input,
  val attentionArrays: List<AugmentedArray<DenseNDArray>>,
  params: LayerParameters,
  activation: ActivationFunction? = SoftmaxBase(),
  dropout: Double = 0.0
) : Layer<InputNDArrayType>(
  inputArray = inputArrays[0],
  inputType = inputType,
  outputArray = AugmentedArray(values = DenseNDArrayFactory.zeros(Shape(inputArrays.first().size))),
  params = params,
  activationFunction = activation,
  dropout = dropout
) {

  /**
   * The helper which executes the forward
   */
  override val forwardHelper = AttentionForwardHelper(layer = this)

  /**
   * The helper which executes the backward
   */
  override val backwardHelper = AttentionBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance
   */
  override val relevanceHelper: RelevanceHelper? = null

  /**
   * The attention mechanism.
   */
  internal val attentionMechanism = AttentionMechanismLayer(
    inputArrays = this.attentionArrays,
    inputType = inputType,
    params = params as AttentionMechanismLayerParameters,
    activation = SoftmaxBase()
  ).apply {
    setParamsErrorsCollector(this@AttentionLayer.getParamsErrorsCollector())
  }

  /**
   * The attention scores.
   */
  val attentionScores: AugmentedArray<DenseNDArray> get() = this.attentionMechanism.outputArray

  /**
   * The attention matrix.
   */
  val attentionMatrix: AugmentedArray<DenseNDArray> get() = this.attentionMechanism.attentionMatrix

  /**
   * Essential requirements.
   */
  init {

    this.params as AttentionMechanismLayerParameters

    require(this.inputArrays.isNotEmpty()) { "The input array cannot be empty." }
    require(this.attentionArrays.isNotEmpty()) { "The attention array cannot be empty." }
    require(this.inputArrays.size == attentionArrays.size) {
      "The input array must have the same length of the attention array."
    }

    this.inputArrays.first().size.let { inputSize ->
      this.inputArrays.forEach { require(it.size == inputSize) }
    }
  }
}
