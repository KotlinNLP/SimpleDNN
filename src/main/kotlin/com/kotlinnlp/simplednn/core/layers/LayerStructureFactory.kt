/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.feedforward.simple.FeedforwardLayerStructure
import com.kotlinnlp.simplednn.core.layers.feedforward.highway.HighwayLayerStructure
import com.kotlinnlp.simplednn.core.layers.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.layers.recurrent.cfn.CFNLayerStructure
import com.kotlinnlp.simplednn.core.layers.recurrent.deltarnn.DeltaRNNLayerStructure
import com.kotlinnlp.simplednn.core.layers.recurrent.gru.GRULayerStructure
import com.kotlinnlp.simplednn.core.layers.recurrent.indrnn.IndRNNLayerStructure
import com.kotlinnlp.simplednn.core.layers.recurrent.lstm.LSTMLayerStructure
import com.kotlinnlp.simplednn.core.layers.recurrent.ran.RANLayerStructure
import com.kotlinnlp.simplednn.core.layers.recurrent.simple.SimpleRecurrentLayerStructure
import com.kotlinnlp.simplednn.core.layers.merge.affine.AffineLayerParameters
import com.kotlinnlp.simplednn.core.layers.merge.affine.AffineLayerStructure
import com.kotlinnlp.simplednn.core.layers.merge.biaffine.BiaffineLayerParameters
import com.kotlinnlp.simplednn.core.layers.merge.biaffine.BiaffineLayerStructure
import com.kotlinnlp.simplednn.core.layers.merge.concat.ConcatLayerParameters
import com.kotlinnlp.simplednn.core.layers.merge.concat.ConcatLayerStructure
import com.kotlinnlp.simplednn.core.layers.merge.product.ProductLayerParameters
import com.kotlinnlp.simplednn.core.layers.merge.product.ProductLayerStructure
import com.kotlinnlp.simplednn.core.layers.merge.sum.SumLayerParameters
import com.kotlinnlp.simplednn.core.layers.merge.sum.SumLayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * Helper that builds a generic [LayerStructure].
 */
object LayerStructureFactory {

  /**
   * Build a new generic [LayerStructure].
   *
   * @param inputArrays the list of input arrays (more then one only for Merge layers)
   * @param outputSize the size of the output array
   * @param params the layer parameters
   * @param connectionType the type of connection from the input to the output
   * @param activationFunction the activation function of the layer
   * @param dropout the probability of dropout (default 0.0). If applying it, the usual value is 0.5 (better 0.25 if
   *                it's the first layer).
   * @param contextWindow the context window in case of recurrent layer, otherwise null
   *
   * @return a new layer structure
   */
  operator fun <InputNDArrayType : NDArray<InputNDArrayType>> invoke(
    inputArrays: List<AugmentedArray<InputNDArrayType>>,
    outputSize: Int,
    params: LayerParameters<*>,
    connectionType: LayerType.Connection,
    activationFunction: ActivationFunction? = null,
    dropout: Double = 0.0,
    contextWindow: LayerContextWindow? = null
  ): LayerStructure<InputNDArrayType> = when (connectionType) {

    LayerType.Connection.Feedforward -> FeedforwardLayerStructure(
      inputArray = inputArrays.first(),
      outputArray = LayerUnit(outputSize),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout)

    LayerType.Connection.Highway -> HighwayLayerStructure(
      inputArray = inputArrays.first(),
      outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(outputSize))),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout)

    LayerType.Connection.Affine -> AffineLayerStructure(
      inputArrays = inputArrays,
      outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(outputSize))),
      params = params as AffineLayerParameters,
      activationFunction = activationFunction,
      dropout = dropout)

    LayerType.Connection.Biaffine -> BiaffineLayerStructure(
      inputArray1 = inputArrays[0],
      inputArray2 = inputArrays[1],
      outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(outputSize))),
      params = params as BiaffineLayerParameters,
      activationFunction = activationFunction,
      dropout = dropout)

    LayerType.Connection.Concat -> ConcatLayerStructure(
      inputArrays = inputArrays,
      outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(outputSize))),
      params = params as ConcatLayerParameters)

    LayerType.Connection.Sum -> SumLayerStructure(
      inputArrays = inputArrays,
      outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(outputSize))),
      params = params as SumLayerParameters)

    LayerType.Connection.Product -> ProductLayerStructure(
      inputArrays = inputArrays,
      outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(outputSize))),
      params = params as ProductLayerParameters)

    LayerType.Connection.SimpleRecurrent -> SimpleRecurrentLayerStructure(
      inputArray = inputArrays.first(),
      outputArray = RecurrentLayerUnit(outputSize),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!)

    LayerType.Connection.GRU -> GRULayerStructure(
      inputArray = inputArrays.first(),
      outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(outputSize))),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!)

    LayerType.Connection.LSTM -> LSTMLayerStructure(
      inputArray = inputArrays.first(),
      outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(outputSize))),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!)

    LayerType.Connection.CFN -> CFNLayerStructure(
      inputArray = inputArrays.first(),
      outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(outputSize))),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!)

    LayerType.Connection.RAN -> RANLayerStructure(
      inputArray = inputArrays.first(),
      outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(outputSize))),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!)

    LayerType.Connection.DeltaRNN -> DeltaRNNLayerStructure(
      inputArray = inputArrays.first(),
      outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(outputSize))),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!)

    LayerType.Connection.IndRNN -> IndRNNLayerStructure(
      inputArray = inputArrays.first(),
      outputArray = LayerUnit(outputSize),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!)
  }
}
