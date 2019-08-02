/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.tpr

import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [TPRLayer] in which the forward is executed
 */
class TPRForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: TPRLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   */
  override fun forward() {

    val params = this.layer.params as TPRLayerParameters
    val x: InputNDArrayType = this.layer.inputArray.values

    this.layer.aR.forward(w = params.wInR.values, b = params.bR.values, x = x)
    this.layer.aS.forward(w = params.wInS.values, b = params.bS.values, x = x)

    this.addRecurrentContribution()

    this.layer.aR.activate()
    this.layer.aS.activate()

    this.layer.r.forward(w = params.r.values, b = null, x = this.layer.aR.values)
    this.layer.s.forward(w = params.s.values, b = null, x = this.layer.aS.values)

    this.layer.bindingMatrix.values.assignValues(a = this.layer.s.values.dot(this.layer.r.values.t))
    this.layer.bindingMatrix.values.vectorize(this.layer.outputArray.values)
  }

  /**
   *
   */
  private fun addRecurrentContribution() {

    val params = this.layer.params as TPRLayerParameters

    this.layer.layerContextWindow.getPrevState()?.let { prevStateLayer ->

      val yPrev: DenseNDArray = prevStateLayer.outputArray.values

      this.layer.aR.values.assignSum(params.wRecR.values.dot(yPrev))
      this.layer.aS.values.assignSum(params.wRecS.values.dot(yPrev))
    }
  }
}
