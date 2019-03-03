/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.deltarnn

import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [DeltaRNNLayer] in which the forward is executed
 */
class DeltaRNNForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: DeltaRNNLayer<InputNDArrayType>
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
    val prevStateLayer: Layer<*>? = this.layer.layerContextWindow.getPrevState()
    val yPrev: DenseNDArray? = prevStateLayer?.outputArray?.values

    wx.assignDot(w, x)

    val c: DenseNDArray = this.calculateCandidate(
      wyRec = if (yPrev != null) this.calculateRecurrentContribution(yPrev) else null)

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
  override fun forward(layerContributions: LayerParameters<*>) {
    this.layer.params as DeltaRNNLayerParameters
    layerContributions as DeltaRNNLayerParameters

    val wx: DenseNDArray = this.layer.wx.values
    var wyRec: DenseNDArray? = null
    val prevStateLayer: Layer<*>? = this.layer.layerContextWindow.getPrevState()
    val yPrev: DenseNDArray? = prevStateLayer?.outputArray?.values

    // w (dot) x
    this.forwardArray(
      contributions = layerContributions.feedforwardUnit.weights.values,
      x = this.layer.inputArray.values,
      y = wx,
      w = this.layer.params.feedforwardUnit.weights.values as DenseNDArray)

    // wRec (dot) yPrev
    if (yPrev != null) {
      wyRec = this.layer.wyRec.values
      this.forwardArray(
        contributions = layerContributions.recurrentUnit.weights.values,
        x =  yPrev,
        y = wyRec,
        w = this.layer.params.recurrentUnit.weights.values as DenseNDArray)
    }

    val c: DenseNDArray = this.calculateCandidateSavingContributions(wyRec = wyRec)
    val p: DenseNDArray = this.calculatePartition()

    val y: DenseNDArray = this.layer.outputArray.values
    y.assignProd(p, c)

    if (yPrev != null) {
      val yRec: DenseNDArray = layerContributions.recurrentUnit.biases.values as DenseNDArray

      yRec.assignValues(p.reverseSub(1.0).assignProd(yPrev))
      y.assignSum(yRec)
    }

    this.layer.outputArray.activate()
  }

  /**
   * Calculate the recurrent contribution as dot product between the output in the previous state and the recurrent
   * weights.
   *
   * @param yPrev the output array in the previous state
   *
   * @return the recurrent contribution
   */
  private fun calculateRecurrentContribution(yPrev: DenseNDArray): DenseNDArray? {
    this.layer.params as DeltaRNNLayerParameters

    val wRec: DenseNDArray = this.layer.params.recurrentUnit.weights.values as DenseNDArray
    val wyRec: DenseNDArray = this.layer.wyRec.values

    return wyRec.assignDot(wRec, yPrev)
  }

  /**
   * Calculate the values of the candidate array.
   *
   *   d1 = beta1 * w (dot) x + beta2 * wRec (dot) yPrev
   *   d2 = alpha * w (dot) x * wRec (dot) yPrev
   *   c = tanh(d1 + d2 + bc)
   *
   * @param wyRec the result of the dot product between wRec and yPrev if any, null otherwise
   *
   * @return the array of candidate
   */
  private fun calculateCandidate(wyRec: DenseNDArray?): DenseNDArray {
    this.layer.params as DeltaRNNLayerParameters

    val c: DenseNDArray = this.layer.candidate.values
    val bc: DenseNDArray = this.layer.params.feedforwardUnit.biases.values as DenseNDArray
    val beta1: DenseNDArray = this.layer.params.beta1.values
    val wx: DenseNDArray = this.layer.wx.values

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
   * Calculate the values of the candidate array, saving its contributions from d1 and d2.
   *
   *   d1 = beta1 * w (dot) x + beta2 * wRec (dot) yPrev
   *   d2 = alpha * w (dot) x * wRec (dot) yPrev
   *   c = tanh(d1 + d2 + bc)
   *
   * @param wyRec the result of the dot product between wRec and yPrev if any, null otherwise
   *
   * @return the array of candidate
   */
  private fun calculateCandidateSavingContributions(wyRec: DenseNDArray?): DenseNDArray {
    this.layer.params as DeltaRNNLayerParameters

    val relevanceSupport = this.layer.relevanceSupport

    val c: DenseNDArray = this.layer.candidate.values
    val bc: DenseNDArray = this.layer.params.feedforwardUnit.biases.values as DenseNDArray
    val beta1: DenseNDArray = this.layer.params.beta1.values
    val wx: DenseNDArray = this.layer.wx.values
    val d1Input: DenseNDArray = relevanceSupport.d1Input.values

    val d1: DenseNDArray =  if (wyRec != null) {
      val beta2: DenseNDArray = this.layer.params.beta2.values
      val halfBc: DenseNDArray = bc.div(2.0) // bc split equally among d1Input and d1Rec
      val d1Rec: DenseNDArray = relevanceSupport.d1Rec.values

      d1Input.assignProd(beta1, wx).assignSum(halfBc) // d1Input is saved to calculate the relevance later
      d1Rec.assignProd(beta2, wyRec).assignSum(halfBc) // d1Rec is saved to calculate the relevance later

      d1Input.sum(d1Rec)

    } else {
      d1Input.assignProd(beta1, wx).assignSum(bc)
    }

    c.assignValues(d1)

    if (wyRec != null) {
      val alpha: DenseNDArray = this.layer.params.alpha.values
      val d2: DenseNDArray = relevanceSupport.d2.values

      d2.assignProd(alpha, wx).assignProd(wyRec) // d2 is saved to calculate the relevance later
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
    val bp: DenseNDArray = this.layer.params.recurrentUnit.biases.values as DenseNDArray
    val p: DenseNDArray = this.layer.partition.values

    p.assignSum(wx, bp)

    this.layer.partition.activate()

    return this.layer.partition.values
  }
}
