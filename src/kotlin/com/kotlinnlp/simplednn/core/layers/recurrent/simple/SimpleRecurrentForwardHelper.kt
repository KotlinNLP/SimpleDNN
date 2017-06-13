/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.simple

import com.kotlinnlp.simplednn.core.layers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [SimpleRecurrentLayerStructure] in which the forward is executed
 */
class SimpleRecurrentForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: SimpleRecurrentLayerStructure<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   * y = f(w (dot) x + b + wRec (dot) yPrev)
   */
  override fun forward() { this.layer.params as SimpleRecurrentLayerParameters

    val w: DenseNDArray = this.layer.params.weights.values as DenseNDArray
    val b: DenseNDArray = this.layer.params.biases.values

    val x: NDArray<*> = this.layer.inputArray.values
    val y: DenseNDArray = this.layer.outputArray.values

    // y = w (dot) x + b
    y.assignDot(w, x).assignSum(b)

    // y += wRec (dot) yAPrev
    val prevStateLayer = this.layer.layerContextWindow.getPrevStateLayer()
    if (prevStateLayer != null) { // recurrent contribute

      val wRec: DenseNDArray = this.layer.params.recurrentWeights.values
      val yPrev: DenseNDArray = prevStateLayer.outputArray.values

      y.assignSum(wRec.dot(yPrev))
    }

    this.layer.outputArray.activate()
  }

  /**
   * Forward the input to the output combining it with the parameters, saving the contributes of the parameters.
   *
   * @param paramsContributes the [LayerParameters] in which to save the contributes of the parameters
   */
  override fun forward(paramsContributes: LayerParameters) {
    TODO("not implemented")
  }
}
