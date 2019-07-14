/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.normalization

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import kotlin.math.sqrt

class NormalizationForwardHelper <InputNDArrayType : NDArray<InputNDArrayType>>(
    override val layer: NormalizationLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Calculate the standard deviation of [arrays]
   *
   * @param meanVector The mean of [arrays]
   * @param arrays The input arrays
   *
   * @return a dense array
   */
  private fun calculateStdDev(meanVector: DenseNDArray, arrays: List<AugmentedArray<InputNDArrayType>>): DenseNDArray{

    val devVector: DenseNDArray = DenseNDArrayFactory.fromNDArray(this.layer.inputArrays[0].values)
    var n = 0.0

    (1 until arrays.size).forEach {
      i ->
      val diffVector: DenseNDArray = DenseNDArrayFactory.fromNDArray(meanVector)
      diffVector.assignSub(this.layer.inputArrays[i].values)
      diffVector.assignProd(diffVector)
      devVector.assignSum(diffVector)
      n += 1
    }

    devVector.assignDiv(n)
    (0 until devVector.length).forEach { i -> devVector[i] = sqrt(devVector[i]) }

    return devVector
  }

  /**
   * Forward the input to the output combining it with the parameters
   */
  private fun calculateMean(arrays: List<AugmentedArray<InputNDArrayType>>): DenseNDArray{
    val meanVector: DenseNDArray = DenseNDArrayFactory.fromNDArray(this.layer.inputArrays[0].values)
    var n = 0.0

    (1 until arrays.size).forEach { i -> meanVector.assignSum(arrays[i].values)
      n += 1
    }

    return meanVector.assignDiv(n)
  }

  /**
   * Forward the input to the output combining it with the parameters
   */
  override fun forward() {

    val meanVector: DenseNDArray = calculateMean(this.layer.inputArrays)

    val devVector: DenseNDArray = calculateStdDev(meanVector, this.layer.inputArrays)

    for ((index, outputArray) in this.layer.outputArrays.withIndex()){
      devVector.assignDiv(this.layer.params.g.values)
      outputArray.values.assignValues(this.layer.inputArrays[index].values as DenseNDArray)
      outputArray.values.assignSub(meanVector)
      outputArray.values.assignProd(devVector).assignSub(this.layer.params.b.values)
    }
  }
}