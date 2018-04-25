/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.indrnn

import com.kotlinnlp.simplednn.core.layers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [IndRNNLayerStructure] in which the forward is executed
 */
class IndRNNForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: IndRNNLayerStructure<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   * y = f(w (dot) x + wRec * yPrev + b)
   */
  override fun forward() { this.layer.params as IndRNNLayerParameters

    // y = w (dot) x + b
    this.layer.outputArray.forward(parameters = this.layer.params.feedforwardUnit, x = this.layer.inputArray.values)

    // y += wRec * yPrev
    this.layer.layerContextWindow.getPrevStateLayer()?.let { prevStateLayer ->

      val wRec = this.layer.params.recurrentWeights.values
      this.layer.outputArray.values.assignSum(wRec.prod(prevStateLayer.outputArray.values))
    }

    this.layer.outputArray.activate()
  }

  /**
   * Forward the input to the output combining it with the parameters, saving the contributions.
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  override fun forward(layerContributions: LayerParameters<*>) {
    TODO("not implemented")
  }
}
