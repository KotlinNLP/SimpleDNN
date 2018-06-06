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
 * The configuration of the (input or output) interface of a layer.
 *
 * @param size size of the unique array of this layer (meaningless if this is the input of a Merge layer)
 * @property sizes the list of sizes of the arrays in this interface
 * @property type the type of the arrays in this interface
 * @property connectionType the type of connection with the interface before (meaningless in case of input interface of
 *                          an input layer)
 * @property activationFunction the activation function (meaningless in case of input interface)
 * @property dropout the probability of dropout (meaningless in case of input interface). If applying it, the usual
 *                   value is 0.5 (better 0.25 if it's the first layer).
 * @property meProp whether to use the 'meProp' errors propagation algorithm (params errors are sparse)
 */
data class LayerInterface(
  val size: Int = -1,
  val sizes: List<Int> = listOf(size),
  val type: LayerType.Input = LayerType.Input.Dense,
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
