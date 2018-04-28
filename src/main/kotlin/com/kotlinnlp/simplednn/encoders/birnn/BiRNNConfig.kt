/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.encoders.birnn

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerType
import java.io.Serializable

/**
 * The configuration of a BiRNN.
 *
 * @param connectionType the recurrent connection type of the BiRNN used to encode tokens
 * @param hiddenActivation the hidden activation function of the BiRNN used to encode tokens
 * @param numberOfLayers number of stacked BiRNNs (default 1)
 */
data class BiRNNConfig(
  val connectionType: LayerType.Connection,
  val hiddenActivation: ActivationFunction?,
  val numberOfLayers: Int = 1
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * @return the string representation of this class
   */
  override fun toString(): String = "%s - %s - %s".format(
    this.connectionType,
    if (this.hiddenActivation != null) this.hiddenActivation::class.simpleName else null,
    this.numberOfLayers
  )
}