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

/**
 * The helper which executes the forward on the [NormLayer].
 */
internal class NormForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: NormLayer<InputNDArrayType>
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

    val dev: DenseNDArray = DenseNDArrayFactory.zeros(this.layer.inputArrays[0].values.shape)
    val e = 0.00000000001

    arrays.indices.forEach { i ->

      val diff: DenseNDArray = DenseNDArrayFactory.fromNDArray(meanVector)

      diff.assignSub(this.layer.inputArrays[i].values)
      diff.assignProd(diff)

      dev.assignSum(diff)
    }

    (0 until dev.length).forEach { i -> dev[i] = sqrt(dev[i] / arrays.size + e) }

    return dev
  }

  /**
   * Forward the input to the output combining it with the parameters
   */
  private fun calculateMean(arrays: List<AugmentedArray<InputNDArrayType>>): DenseNDArray{

    val mean: DenseNDArray = DenseNDArrayFactory.zeros(this.layer.inputArrays[0].values.shape)

    arrays.indices.forEach { i ->
      mean.assignSum(arrays[i].values)
    }

    return mean.assignDiv(arrays.size.toDouble())
  }

  /**
   * Forward the input to the output combining it with the parameters
   */
  override fun forward() {

    val mean: DenseNDArray = calculateMean(this.layer.inputArrays)
    val dev: DenseNDArray = calculateStdDev(mean, this.layer.inputArrays)

    this.layer.devStdArray.assignValues(dev)
    dev.assignValues(this.layer.params.g.values.div(dev))

    this.layer.meanArray.assignValues(mean)

    this.layer.outputArrays.forEachIndexed { index, outputArray ->

      outputArray.valuesNotActivated.assignValues(this.layer.inputArrays[index].values as DenseNDArray)
      outputArray.valuesNotActivated.assignSub(mean)
      outputArray.valuesNotActivated.assignProd(dev).assignSum(this.layer.params.b.values)

      outputArray.activate()
    }
  }
}
