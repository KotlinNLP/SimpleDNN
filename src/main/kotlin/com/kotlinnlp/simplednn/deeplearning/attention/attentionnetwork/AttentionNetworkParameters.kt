/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attention.attentionnetwork

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.layers.models.attention.attentionmechanism.AttentionMechanismLayerParameters
import java.io.Serializable

/**
 * The parameters of the Attention Network.
 *
 * @property inputSize the size of the input arrays
 * @property attentionSize the size of the attention arrays
 * @param sparseInput `true` if the input arrays are sparse, `false` if they are dense
 * @param weightsInitializer the initializer of the transform weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the transform biases (zeros if null, default: Glorot)
 */
class AttentionNetworkParameters(
  val inputSize: Int,
  val attentionSize: Int,
  sparseInput: Boolean = false,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer()
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The size of the output array.
   */
  val outputSize: Int = this.inputSize

  /**
   * The parameters of the transform network.
   */
  val transform = StackedLayersParameters(
    LayerInterface(
      size = this.inputSize,
      type = if (sparseInput) LayerType.Input.Sparse else LayerType.Input.Dense),
    LayerInterface(
      size = this.attentionSize,
      connectionType = LayerType.Connection.Feedforward,
      activationFunction = Tanh),
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)

  /**
   * The parameters of the attention mechanism.
   */
  val attention = AttentionMechanismLayerParameters(
    inputSize = this.attentionSize,
    weightsInitializer = weightsInitializer)
}
