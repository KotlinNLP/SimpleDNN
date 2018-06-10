/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.birnn.utils

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.types.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object BiRNNEncoderUtils {

  /**
   *
   */
  fun buildInputSequence(): List<DenseNDArray> = listOf(
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, 0.6)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, -0.4)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, -0.7))
  )

  /**
   *
   */
  fun buildOutputErrorsSequence(): List<DenseNDArray> = listOf(
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, -0.8, 0.1, 0.4, 0.6, -0.4)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(0.6, 0.6, 0.7, 0.7, -0.6, 0.3)),
    DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.1, -0.1, 0.1, -0.8, 0.4, -0.5))
  )

  /**
   *
   */
  fun buildBiRNN(): BiRNN {

    val birnn = BiRNN(
      inputSize = 2,
      inputType = LayerType.Input.Dense,
      hiddenSize = 3,
      hiddenActivation = Tanh(),
      recurrentConnectionType = LayerType.Connection.SimpleRecurrent
    )

    this.initL2RParameters(params = birnn.leftToRightNetwork.model.paramsPerLayer[0] as SimpleRecurrentLayerParameters)
    this.initR2LParameters(params = birnn.rightToLeftNetwork.model.paramsPerLayer[0] as SimpleRecurrentLayerParameters)

    return birnn
  }

  /**
   *
   */
  private fun initL2RParameters(params: SimpleRecurrentLayerParameters) {

    params.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(-0.9, 0.4),
      doubleArrayOf(0.7, -1.0),
      doubleArrayOf(-0.9, -0.4)
    )))

    params.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, -0.3, 0.8)))

    params.unit.recurrentWeights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(0.1, 0.9, -0.5),
      doubleArrayOf(-0.6, 0.7, 0.7),
      doubleArrayOf(0.3, 0.9, 0.0)
    )))
  }

  /**
   *
   */
  private fun initR2LParameters(params: SimpleRecurrentLayerParameters) {

    params.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(0.3, 0.1),
      doubleArrayOf(0.6, 0.0),
      doubleArrayOf(-0.7, 0.1)
    )))

    params.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.2, -0.9, -0.2)))

    params.unit.recurrentWeights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      doubleArrayOf(-0.2, 0.7, 0.7),
      doubleArrayOf(-0.2, 0.0, -1.0),
      doubleArrayOf(0.5, -0.4, 0.4)
    )))
  }
}
