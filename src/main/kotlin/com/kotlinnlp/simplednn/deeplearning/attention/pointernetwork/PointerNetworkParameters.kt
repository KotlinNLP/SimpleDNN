/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attention.pointernetwork

import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.layers.models.attention.attentionmechanism.AttentionMechanismLayerParameters
import java.io.Serializable

/**
 * The parameters of the [PointerNetworkProcessor].
 *
 * @property mergeParams the parameters of the merge network
 * @property attentionParams the parameters of the attention structure
 */
class PointerNetworkParameters(
  val mergeParams: StackedLayersParameters,
  val attentionParams: AttentionMechanismLayerParameters
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }
}
