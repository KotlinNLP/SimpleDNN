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
 * @property meProp whether to use the 'meProp' errors propagation algorithm (params errors are sparse)
 */
data class LayerConfiguration(
  val size: Int,
  val inputType: LayerType.Input = LayerType.Input.Dense,
  val connectionType: LayerType.Connection? = null,
  val activationFunction: ActivationFunction? = null,
  val meProp: Boolean = false,
  val dropout: Double = 0.0
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }
}
