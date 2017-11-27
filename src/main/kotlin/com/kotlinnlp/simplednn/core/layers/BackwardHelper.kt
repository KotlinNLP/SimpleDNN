/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArrayMask
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import java.util.*

/**
 * The helper which executes the backward on a [layer].
 */
interface BackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>> {

  /**
   * The [LayerStructure] in which the backward is executed.
   */
  val layer: LayerStructure<InputNDArrayType>

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param paramsErrors the errors of the parameters which will be filled
   * @param propagateToInput whether to propagate the errors to the input array
   * @param mePropK the k factor of the 'meProp' algorithm to propagate from top k (in percentage) output nodes
   *                (ignored if null)
   */
  fun backward(paramsErrors: LayerParameters<*>, propagateToInput: Boolean = false, mePropK: Double?)

  /**
   * @param mePropK the k factor of the 'meProp' algorithm (the percentage of the top k nodes)
   *
   * @return the mask of the top k ([mePropK]) elements of the output array.
   */
  fun getOutputMask(mePropK: Double): NDArrayMask {

    val y: DenseNDArray = this.layer.outputArray.values
    val nTopElements: Int = Math.round(mePropK * y.length).toInt()

    val minHeap = PriorityQueue<Pair<Int, Double>>(y.length, Comparator({ a, b ->
      val diff: Double = a.second - b.second
      if (diff > 0.0) -1 else if (diff < 0.0) 1 else 0
    }))

    (0 until y.length).forEach { i -> minHeap.add(Pair(i, y[i])) }

    return NDArrayMask(
      dim1 = Array(size = nTopElements, init = { minHeap.remove().first }),
      dim2 = Array(size = nTopElements, init = { 0 }))
  }
}
