/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.feedforward.simple

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArrayFactory

/**
 *
 */
internal object FeedforwardLayerStructureUtils {

  /**
   *
   */
  fun buildLayer45(): FeedforwardLayer<DenseNDArray> = FeedforwardLayer(
    inputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.9, -0.9, 1.0))),
    inputType = LayerType.Input.Dense,
    outputArray = AugmentedArray.zeros(5),
    params = getParams45(),
    activationFunction = Tanh,
    dropout = 0.0
  )

  /**
   *
   */
  fun getParams45() = FeedforwardLayerParameters(inputSize = 4, outputSize = 5).apply {

    unit.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.5, 0.6, -0.8, -0.6),
        doubleArrayOf(0.7, -0.4, 0.1, -0.8),
        doubleArrayOf(0.7, -0.7, 0.3, 0.5),
        doubleArrayOf(0.8, -0.9, 0.0, -0.1),
        doubleArrayOf(0.4, 1.0, -0.7, 0.8)
      )))

    unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.0, -0.3, 0.8, -0.4)))
  }

  /**
   *
   */
  fun getOutputGold5(): DenseNDArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.5, -0.4, -0.9, 0.9))

  /**
   *
   */
  fun buildLayer53(): FeedforwardLayer<DenseNDArray> = FeedforwardLayer(
    inputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, -0.8, 0.0, 0.7, -0.2))),
    inputType = LayerType.Input.Dense,
    outputArray = AugmentedArray.zeros(3),
    params = getParams53(),
    activationFunction = Softmax(),
    dropout = 0.0
  )

  /**
   *
   */
  fun buildLayer53SparseBinary(): FeedforwardLayer<SparseBinaryNDArray> {

    val input: SparseBinaryNDArray = SparseBinaryNDArrayFactory.arrayOf(activeIndices = listOf(2, 4), shape = Shape(5))

    return FeedforwardLayer(
      inputArray = AugmentedArray(input).apply { setActivation(Tanh) },
      inputType = LayerType.Input.SparseBinary,
      outputArray = AugmentedArray.zeros(3),
      params = getParams53(),
      activationFunction = Softmax(),
      dropout = 0.0)
  }

  /**
   *
   */
  fun getParams53() = FeedforwardLayerParameters(inputSize = 5, outputSize = 3).apply {

    unit.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.8, -0.8, 0.9, -1.0, -0.1),
        doubleArrayOf(0.9, 0.6, 0.7, 0.6, 0.6),
        doubleArrayOf(-0.1, 0.0, 0.3, 0.0, 0.3)
      )))

    unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.5, 0.1, 0.2)))
  }

  /**
   *
   */
  fun getOutputGold3(): DenseNDArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(1.0, 0.0, 0.0))
}
