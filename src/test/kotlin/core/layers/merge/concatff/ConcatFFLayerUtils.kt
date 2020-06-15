/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.merge.concatff

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.merge.concatff.ConcatFFLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.concatff.ConcatFFLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal object ConcatFFLayerUtils {

  /**
   *
   */
  fun buildLayer(): ConcatFFLayer<DenseNDArray> = ConcatFFLayer(
    inputArrays = listOf(
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.9, 0.9, 0.6, 0.1))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.5))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.7, -0.7, 0.8)))
    ),
    inputType = LayerType.Input.Dense,
    outputArray = AugmentedArray.zeros(3),
    params = buildParams(),
    activationFunction = Tanh,
    dropout = 0.0
  )

  /**
   *
   */
  fun getOutputErrors() = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.7, -0.8, 0.6))

  /**
   *
   */
  private fun buildParams() = ConcatFFLayerParameters(inputsSize = listOf(4, 2, 3), outputSize = 3).apply {

    output.unit.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(-0.1, -0.3, 0.5, 0.6, -0.6, 0.6, 0.4, -0.2, -0.9),
        doubleArrayOf(0.6, 0.6, -0.2, 0.3, 0.7, -0.2, 0.9, -0.3, -0.5),
        doubleArrayOf(0.7, 0.7, 0.0, -0.1, -0.9, 0.4, 0.2, 0.1, -0.4)
      ))
    )

    output.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.2, -0.7)))
  }
}
