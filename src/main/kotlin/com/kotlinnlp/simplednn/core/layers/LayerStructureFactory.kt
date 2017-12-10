/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerStructure
import com.kotlinnlp.simplednn.core.layers.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.layers.recurrent.cfn.CFNLayerStructure
import com.kotlinnlp.simplednn.core.layers.recurrent.deltarnn.DeltaRNNLayerStructure
import com.kotlinnlp.simplednn.core.layers.recurrent.gru.GRULayerStructure
import com.kotlinnlp.simplednn.core.layers.recurrent.lstm.LSTMLayerStructure
import com.kotlinnlp.simplednn.core.layers.recurrent.ran.RANLayerStructure
import com.kotlinnlp.simplednn.core.layers.recurrent.simple.SimpleRecurrentLayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object LayerStructureFactory {

  operator fun <InputNDArrayType : NDArray<InputNDArrayType>> invoke(
    inputArray: AugmentedArray<InputNDArrayType>,
    outputSize: Int,
    params: LayerParameters<*>,
    activationFunction: ActivationFunction?,
    connectionType: LayerType.Connection,
    dropout: Double = 0.0,
    contextWindow: LayerContextWindow? = null): LayerStructure<InputNDArrayType> = when(connectionType) {

    LayerType.Connection.Feedforward -> FeedforwardLayerStructure(
      inputArray = inputArray,
      outputArray = LayerUnit(outputSize),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout
    )

    LayerType.Connection.SimpleRecurrent -> SimpleRecurrentLayerStructure(
      inputArray = inputArray,
      outputArray = RecurrentLayerUnit(outputSize),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!
    )

    LayerType.Connection.GRU -> GRULayerStructure(
      inputArray = inputArray,
      outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(outputSize))),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!
    )

    LayerType.Connection.LSTM -> LSTMLayerStructure(
      inputArray = inputArray,
      outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(outputSize))),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!
    )

    LayerType.Connection.CFN -> CFNLayerStructure(
      inputArray = inputArray,
      outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(outputSize))),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!
    )

    LayerType.Connection.RAN -> RANLayerStructure(
      inputArray = inputArray,
      outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(outputSize))),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!
    )

    LayerType.Connection.DeltaRNN -> DeltaRNNLayerStructure(
      inputArray = inputArray,
      outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(outputSize))),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!
    )
  }
}
