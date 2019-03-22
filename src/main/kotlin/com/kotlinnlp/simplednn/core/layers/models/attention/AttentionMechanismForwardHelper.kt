/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.attention

import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [AttentionMechanismLayer] in which the forward is executed
 */
class AttentionMechanismForwardHelper(
  override val layer: AttentionMechanismLayer
) : ForwardHelper<DenseNDArray>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   *    am = attention matrix
   *    cv = context vector
   *
   *    y = activation(am (dot) cv)
   */
  override fun forward() { this.layer.params as AttentionMechanismLayerParameters

    this.layer.outputArray.assignValues(this.layer.attentionMatrix.values.dot(this.layer.params.contextVector.values))
    this.layer.outputArray.activate()
  }

  /**
   * Forward the input to the output combining it with the parameters, saving the contributions.
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  override fun forward(layerContributions: LayerParameters<*>) {
    throw NotImplementedError("Forward with contributions not available for the Attention layer.")
  }
}
