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
import com.kotlinnlp.simplednn.core.layers.LayerUnit
import com.kotlinnlp.simplednn.core.layers.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.feedforward.simple.FeedforwardLayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArrayFactory

/**
 *
 */
object FeedforwardLayerStructureUtils {

  /**
   *
   */
  fun buildLayer45(): FeedforwardLayerStructure<DenseNDArray> {

    return FeedforwardLayerStructure(
      inputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.9, -0.9, 1.0))),
      outputArray = LayerUnit(5),
      params = getParams45(),
      activationFunction = Tanh())
  }

  /**
   *
   */
  fun getParams45(): FeedforwardLayerParameters {

    val params = FeedforwardLayerParameters(inputSize = 4, outputSize = 5)

    params.unit.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.5, 0.6, -0.8, -0.6),
        doubleArrayOf(0.7, -0.4, 0.1, -0.8),
        doubleArrayOf(0.7, -0.7, 0.3, 0.5),
        doubleArrayOf(0.8, -0.9, 0.0, -0.1),
        doubleArrayOf(0.4, 1.0, -0.7, 0.8)
      )))

    params.unit.biases.values.assignValues(
      DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.0, -0.3, 0.8, -0.4))
    )

    return params
  }

  /**
   *
   */
  fun getOutputGold5(): DenseNDArray =  DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.5, -0.4, -0.9, 0.9))

  /**
   *
   */
  fun buildLayer53(): FeedforwardLayerStructure<DenseNDArray> {

    val inputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, -0.8, 0.0, 0.7, -0.2)))

    return FeedforwardLayerStructure(
      inputArray = inputArray,
      outputArray = LayerUnit(3),
      params = getParams53(),
      activationFunction = Softmax())
  }

  /**
   *
   */
  fun buildLayer53SparseBinary(): FeedforwardLayerStructure<SparseBinaryNDArray> {

    val inputArray = AugmentedArray(SparseBinaryNDArrayFactory.arrayOf(
      activeIndices = listOf(2, 4),
      shape = Shape(5)))
    inputArray.setActivation(Tanh())

    return FeedforwardLayerStructure(
      inputArray = inputArray,
      outputArray = LayerUnit(3),
      params = getParams53(),
      activationFunction = Softmax())
  }

  /**
   *
   */
  fun getParams53(): FeedforwardLayerParameters {

    val params = FeedforwardLayerParameters(inputSize = 5, outputSize = 3)

    params.unit.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.8, -0.8, 0.9, -1.0, -0.1),
        doubleArrayOf(0.9, 0.6, 0.7, 0.6, 0.6),
        doubleArrayOf(-0.1, 0.0, 0.3, 0.0, 0.3)
      )))

    params.unit.biases.values.assignValues(
      DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.5, 0.1, 0.2)))

    return params
  }

  /**
   *
   */
  fun getOutputGold3(): DenseNDArray =  DenseNDArrayFactory.arrayOf(doubleArrayOf(1.0, 0.0, 0.0))
}
