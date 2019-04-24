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
 * @property sizes the list of sizes of the arrays in this interface
 * @property type the type of the arrays in this interface
 * @property connectionType the type of connection with the interface before (meaningless in case of input interface)
 * @property activationFunction the activation function (meaningless if this is an input interface)
 * @property dropout the probability of dropout (meaningless in case of input interface). If applying it, the usual
 *                   value is 0.5 (better 0.25 if it's the first layer).
 */
data class LayerInterface(
  val sizes: List<Int>,
  val type: LayerType.Input = LayerType.Input.Dense,
  val connectionType: LayerType.Connection? = null,
  val activationFunction: ActivationFunction? = null,
  val dropout: Double = 0.0
) : Serializable {

  /**
   * Build a [LayerInterface] with a unique array (not the input of a Merge layer).
   *
   * @param size the size of the unique array of this interface
   * @param type the type of the arrays in this interface
   * @param connectionType the type of connection with the interface before (meaningless in case of input interface)
   * @param activationFunction the activation function (meaningless if this is an input interface)
   * @param dropout the probability of dropout (meaningless in case of input interface). If applying it, the usual
   *                value is 0.5 (better 0.25 if it's the first layer).
   */
  constructor(
    size: Int,
    type: LayerType.Input = LayerType.Input.Dense,
    connectionType: LayerType.Connection? = null,
    activationFunction: ActivationFunction? = null,
    dropout: Double = 0.0
  ): this(
    sizes = listOf(size),
    type = type,
    connectionType = connectionType,
    activationFunction = activationFunction,
    dropout = dropout
  )

  /**
   * The size of the unique array of this interface (meaningless in case of input interface of a Merge layer).
   */
  val size: Int = if (sizes.size == 1) sizes.first() else -1

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }
}
