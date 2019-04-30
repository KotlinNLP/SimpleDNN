/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.ltm

import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [LTMLayer] in which the forward is executed
 */
class LTMForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: LTMLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   * y = cell * l3
   */
  override fun forward() {

    val prevStateLayer: LTMLayer<*>? = this.layer.layerContextWindow.getPrevState() as? LTMLayer<*>

    this.forwardInputGates(prevStateLayer)
    this.forwardCell(prevStateLayer)

    val l3: DenseNDArray = this.layer.inputGate3.values
    val cell: DenseNDArray = this.layer.cell.values
    val y: DenseNDArray = this.layer.outputArray.values

    y.assignProd(cell, l3)
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
   * Input gates forward.
   *
   * l1 = sigmoid(w1 (dot) (x + yPrev))
   * l2 = sigmoid(w2 (dot) (x + yPrev))
   * l3 = sigmoid(w3 (dot) (x + yPrev))
   *
   * @param prevStateLayer the layer in the previous state
   */
  private fun forwardInputGates(prevStateLayer: Layer<*>?) {

    this.layer.params as LTMLayerParameters

    val yPrev: DenseNDArray? = prevStateLayer?.outputArray?.values
    this.layer.x = yPrev?.sum(this.layer.inputArray.values) ?: this.layer.inputArray.values

    this.layer.inputGate1.forward(w = this.layer.params.inputGate1.weights.values, b = null, x = this.layer.x)
    this.layer.inputGate2.forward(w = this.layer.params.inputGate2.weights.values, b = null, x = this.layer.x)
    this.layer.inputGate3.forward(w = this.layer.params.inputGate3.weights.values, b = null, x = this.layer.x)

    this.layer.inputGate1.activate()
    this.layer.inputGate2.activate()
    this.layer.inputGate3.activate()
  }

  /**
   * Cell forward.
   *
   * c = l1 * l2 + cellPrev
   * cell = sigmoid(c (dot) wCell + bCell)
   *
   * @param prevStateLayer the layer in the previous state
   */
  private fun forwardCell(prevStateLayer: Layer<*>?) {

    this.layer.params as LTMLayerParameters

    val l1: DenseNDArray = this.layer.inputGate1.values
    val l2: DenseNDArray = this.layer.inputGate2.values
    val cellPrev: DenseNDArray? = (prevStateLayer as? LTMLayer<*>)?.cell?.values

    this.layer.c.assignValuesByProd(l1, l2)
    if (cellPrev != null) this.layer.c.values.assignSum(cellPrev)

    this.layer.cell.forward(w = this.layer.params.cell.weights.values, b = null, x = this.layer.c.values)
    this.layer.cell.activate()
  }
}
