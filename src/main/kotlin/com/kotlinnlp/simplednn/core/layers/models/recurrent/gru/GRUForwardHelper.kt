/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.gru

import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.LayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [GRULayerStructure] in which the forward is executed
 */
class GRUForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: GRULayerStructure<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   * y = p * c + (1 - p) * yPrev
   */
  override fun forward() {

    val prevStateLayer = this.layer.layerContextWindow.getPrevStateLayer()

    this.setGates(prevStateLayer) // must be called before accessing to the activated values of the gates

    val y: DenseNDArray = this.layer.outputArray.values
    val c: DenseNDArray = this.layer.candidate.values
    val p: DenseNDArray = this.layer.partitionGate.values

    // y = p * c
    y.assignProd(p, c)

    // y += (1 - p) * yPrev
    if (prevStateLayer != null) {
      val yPrev: DenseNDArray = prevStateLayer.outputArray.values
      y.assignSum(p.reverseSub(1.0).prod(yPrev))
    }
  }

  /**
   * Forward the input to the output combining it with the parameters, saving the contributions.
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  override fun forward(layerContributions: LayerParameters<*>) {
    TODO("not implemented")
  }

  /**
   * Set gates values
   *
   * r = sigmoid(wr (dot) x + br + wrRec (dot) yPrev)
   * p = sigmoid(wp (dot) x + bp + wpRec (dot) yPrev)
   * c = f(wc (dot) x + bc + wcRec (dot) (yPrev * r))
   */
  private fun setGates(prevStateLayer: LayerStructure<*>?) { this.layer.params as GRULayerParameters

    val x: InputNDArrayType = this.layer.inputArray.values

    this.layer.resetGate.forward(this.layer.params.resetGate, x)
    this.layer.partitionGate.forward(this.layer.params.partitionGate, x)
    this.layer.candidate.forward(this.layer.params.candidate, x)

    if (prevStateLayer != null) { // recurrent contribution for r and p
      val yPrev = prevStateLayer.outputArray.values
      this.layer.resetGate.addRecurrentContribution(this.layer.params.resetGate, yPrev)
      this.layer.partitionGate.addRecurrentContribution(this.layer.params.partitionGate, yPrev)
    }

    this.layer.resetGate.activate()
    this.layer.partitionGate.activate()

    if (prevStateLayer != null) { // recurrent contribution for c
      val yPrev = prevStateLayer.outputArray.values
      val r = this.layer.resetGate.values
      this.layer.candidate.addRecurrentContribution(this.layer.params.candidate, r.prod(yPrev))
    }

    this.layer.candidate.activate()
  }
}
