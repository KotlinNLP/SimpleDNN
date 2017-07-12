/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.deltarnn

import com.kotlinnlp.simplednn.core.layers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.LayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [DeltaRNNLayerStructure] in which the forward is executed
 */
class DeltaRNNForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: DeltaRNNLayerStructure<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   *   y = f(p * c + (1 - p) * yPrev)
   */
  override fun forward() { this.layer.params as DeltaRNNLayerParameters

    val x: InputNDArrayType = this.layer.inputArray.values
    val w: DenseNDArray = this.layer.params.feedforwardUnit.weights.values as DenseNDArray
    val wx: DenseNDArray = this.layer.wx.values
    val prevStateLayer: LayerStructure<*>? = this.layer.layerContextWindow.getPrevStateLayer()
    val yPrev: DenseNDArray? = prevStateLayer?.outputArray?.values

    wx.assignDot(w, x)

    val c: DenseNDArray = this.calculateCandidate(yPrev = yPrev)
    val p: DenseNDArray = this.calculatePartition()

    val y: DenseNDArray = this.layer.outputArray.values
    y.assignProd(p, c)

    if (yPrev != null) {
      y.assignSum(p.reverseSub(1.0).assignProd(yPrev))
    }

    this.layer.outputArray.activate()
  }

  /**
   * Forward the input to the output combining it with the parameters, saving the contributions.
   *
   * @param layerContributions the structure in which to save the contributions during the calculations
   */
  override fun forward(layerContributions: LayerParameters) {
  }

  /**
   * Calculate the recurrent contribution as dot product between the output in the previous state and the recurrent
   * weights.
   *
   * @return the recurrent contribution (can be null)
   */
  private fun calculateRecurrentContribution(yPrev: DenseNDArray?): DenseNDArray? {
    this.layer.params as DeltaRNNLayerParameters

    return if (yPrev != null) {
      val wRec: DenseNDArray = this.layer.params.recurrentUnit.weights.values as DenseNDArray
      val wyRec: DenseNDArray = this.layer.wyRec.values

      return wyRec.assignDot(wRec, yPrev)

    } else {
      null
    }
  }

  /**
   * Calculate the values of the candidate array.
   *
   *   d1 = beta1 * w (dot) x + beta2 * wRec (dot) yPrev
   *   d2 = alpha * w (dot) x * wRec (dot) yPrev
   *   c = tanh(d1 + d2 + bc)
   *
   * @param yPrev the output array in the previous state
   *
   * @return the array of candidate
   */
  private fun calculateCandidate(yPrev: DenseNDArray?): DenseNDArray {
    this.layer.params as DeltaRNNLayerParameters

    val c: DenseNDArray = this.layer.candidate.values
    val bc: DenseNDArray = this.layer.params.feedforwardUnit.biases.values
    val beta1: DenseNDArray = this.layer.params.beta1.values
    val wx: DenseNDArray = this.layer.wx.values
    val wyRec: DenseNDArray? = this.calculateRecurrentContribution(yPrev)

    val d1: DenseNDArray = beta1.prod(wx)

    if (wyRec != null) {
      val beta2: DenseNDArray = this.layer.params.beta2.values
      d1.assignSum(beta2.prod(wyRec))
    }

    c.assignSum(d1, bc)

    if (wyRec != null) {
      val alpha: DenseNDArray = this.layer.params.alpha.values
      val d2: DenseNDArray = alpha.prod(wx).assignProd(wyRec)
      c.assignSum(d2)
    }

    this.layer.candidate.activate()

    return c
  }

  /**
   * Calculate the values of the partition array.
   *
   *   p = sigmoid(w (dot) x + bp)
   *
   * @return the array of partition
   */
  private fun calculatePartition(): DenseNDArray { this.layer.params as DeltaRNNLayerParameters

    val wx: DenseNDArray = this.layer.wx.values
    val bp: DenseNDArray = this.layer.params.recurrentUnit.biases.values
    val p: DenseNDArray = this.layer.partition.values

    p.assignSum(wx, bp)

    this.layer.partition.activate()

    return this.layer.partition.values
  }
}
