/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import java.io.Serializable
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction

/**
 * The configuration of a Layer.
 *
 * @param size size of the unique array of this layer (meaningless if this is the input of a Merge layer)
 * @param sizes the list of sizes of the arrays in this layer
 * @param inputType the type of the arrays in this layer
 * @param connectionType the type of connection with the layer before (meaningless in case of first layer)
 * @param activationFunction the activation function
 * @param dropout the probability of dropout (default 0.0). If applying it, the usual value is 0.5 (better 0.25 if
 *                it's the first layer).
 * @property meProp whether to use the 'meProp' errors propagation algorithm (params errors are sparse)
 */
data class LayerConfiguration(
  val size: Int = -1,
  val sizes: List<Int> = listOf(size),
  val inputType: LayerType.Input = LayerType.Input.Dense,
  val connectionType: LayerType.Connection? = null,
  val activationFunction: ActivationFunction? = null,
  val meProp: Boolean = false,
  val dropout: Double = 0.0
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }
}
