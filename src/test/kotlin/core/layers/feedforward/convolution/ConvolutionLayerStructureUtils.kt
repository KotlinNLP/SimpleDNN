/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.feedforward.convolution

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.feedforward.convolution.ConvolutionLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.feedforward.convolution.ConvolutionLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 *
 */
object ConvolutionLayerStructureUtils {
  /**
   *
   */
  private fun buildarray1(): DenseNDArray {

    return DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.4, 0.1, -0.9, -0.5),
        doubleArrayOf(-0.4, 0.3, 0.7, -0.3),
        doubleArrayOf(0.8, 0.2, 0.6, 0.7),
        doubleArrayOf(0.2, -0.1, 0.6, -0.2)))
  }
  /**
   *
   */
  private fun buildarray2(): DenseNDArray {

    return DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.3, -0.5, 0.4, -0.2),
        doubleArrayOf(0.1, -0.5, 0.3, 0.7),
        doubleArrayOf(0.7, 0.0, 0.2, 0.6),
        doubleArrayOf(0.4, 0.3, -0.9, 0.3)))
  }
  /**
   *
   */
  private fun buildarray3(): DenseNDArray {

    return DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.4, 0.5, -0.9, 0.0),
        doubleArrayOf(-0.3, -0.8, 0.1, 0.3),
        doubleArrayOf(-0.2, 0.7, 0.0, 0.7),
        doubleArrayOf(-0.6, 0.4, -0.1, 0.6)))
  }
  /**
   *
   */
  fun getOutputGold1(): DenseNDArray {

    return DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(1.0, 0.0, 0.0),
        doubleArrayOf(1.0, 0.0, 1.0),
        doubleArrayOf(1.0, 0.0, 0.0)))
  }
  /**
   *
   */
  fun getOutputGold2(): DenseNDArray {

    return DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.0, 0.0, 1.0),
        doubleArrayOf(0.0, 1.0, 0.0),
        doubleArrayOf(1.0, 0.0, 0.0)))
  }

  /**
   *
   */
  fun getOutputGold3(): DenseNDArray {

    return DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(1.0, 0.0),
        doubleArrayOf(1.0, 0.0)))
  }
  /**
   *
   */
  fun getOutputGold4(): DenseNDArray {

    return DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.0, 0.0),
        doubleArrayOf(0.0, 1.0)))
  }


  /**
   *
   */
  fun buildLayer443(): ConvolutionLayer<DenseNDArray> {

    return ConvolutionLayer(
        inputArrays = listOf(
            AugmentedArray(values = buildarray1()),
            AugmentedArray(values = buildarray2()),
            AugmentedArray(values = buildarray3())),
        inputSize = Shape(4,4),
        inputType = LayerType.Input.Dense,
        params = getParams333(),
        activationFunction = Tanh())
  }

  /**
   *
   */
  fun buildLayer443str2(): ConvolutionLayer<DenseNDArray> {

    return ConvolutionLayer(
        inputArrays = listOf(
            AugmentedArray(values = buildarray1()),
            AugmentedArray(values = buildarray2()),
            AugmentedArray(values = buildarray3())),
        inputSize = Shape(4,4),
        inputType = LayerType.Input.Dense,
        xStride = 2,
        yStride = 2,
        params = getParams333(),
        activationFunction = Tanh())
  }

  /**
   *
   */
  private fun getParams333(): ConvolutionLayerParameters {

    val params = ConvolutionLayerParameters(
        kernelSize = Shape(2,2),
        inputChannels = 3,
        outputChannels = 2)

    params.paramsList[0].values.assignValues(
        DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.5, 0.3),
            doubleArrayOf(0.0, -0.1))))

    params.paramsList[1].values.assignValues(
        DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.4, -0.1),
            doubleArrayOf(0.1, 0.9))))

    params.paramsList[2].values.assignValues(
        DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.6, 0.4),
            doubleArrayOf(0.5, 0.2))))

    params.paramsList[3].values.assignValues(
        DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.2, 0.7),
            doubleArrayOf(0.4, -0.9))))

    params.paramsList[4].values.assignValues(
        DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.0, 0.0),
            doubleArrayOf(0.1, 0.9))))

    params.paramsList[5].values.assignValues(
        DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.3, 0.4),
            doubleArrayOf(-0.4, 0.4))))

    params.paramsList[6].values.assignValues(
        DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.1)))

    params.paramsList[7].values.assignValues(
        DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1)))

    return params
  }

}