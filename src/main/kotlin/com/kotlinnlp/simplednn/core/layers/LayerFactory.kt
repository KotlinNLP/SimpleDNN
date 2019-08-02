/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.highway.HighwayLayer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.squareddistance.SquaredDistanceLayer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.squareddistance.SquaredDistanceLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayerContextWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.cfn.CFNLayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.deltarnn.DeltaRNNLayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.gru.GRULayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.indrnn.IndRNNLayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.lstm.LSTMLayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.ran.RANLayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.simple.SimpleRecurrentLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.affine.AffineLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.affine.AffineLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.avg.AvgLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.avg.AvgLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.biaffine.BiaffineLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.biaffine.BiaffineLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.concat.ConcatLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.concat.ConcatLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.product.ProductLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.product.ProductLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.sub.SubLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.sub.SubLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.sum.SumLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.sum.SumLayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.RecurrentLayerUnit
import com.kotlinnlp.simplednn.core.layers.models.recurrent.ltm.LTMLayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.tpr.TPRLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray

/**
 * Helper that builds a generic [Layer].
 */
object LayerFactory {

  /**
   * Base layer factory, given input and output configurations.
   *
   * @param inputConfiguration layers layersConfiguration of the input array
   * @param outputConfiguration the layersConfiguration of the output array
   * @param params the network parameters of the current layer
   *
   * @return a new Layer
   */
  operator fun invoke(inputConfiguration: LayerInterface,
                      outputConfiguration: LayerInterface,
                      params: LayerParameters,
                      contextWindow: LayerContextWindow? = null): Layer<*> {

    require(outputConfiguration.connectionType != null) {
      "Output layer configurations must have a not null connectionType"
    }

    return when (inputConfiguration.type) {

      LayerType.Input.Dense -> LayerFactory(
        inputArrays = inputConfiguration.sizes.map { AugmentedArray<DenseNDArray>(size = it) },
        inputType = inputConfiguration.type,
        outputSize = outputConfiguration.size,
        params = params,
        activationFunction = outputConfiguration.activationFunction,
        connectionType = outputConfiguration.connectionType,
        dropout = inputConfiguration.dropout,
        contextWindow = contextWindow)

      LayerType.Input.Sparse -> LayerFactory(
        inputArrays = inputConfiguration.sizes.map { AugmentedArray<SparseNDArray>(size = it) },
        inputType = inputConfiguration.type,
        outputSize = outputConfiguration.size,
        params = params,
        activationFunction = outputConfiguration.activationFunction,
        connectionType = outputConfiguration.connectionType,
        dropout = inputConfiguration.dropout,
        contextWindow = contextWindow)

      LayerType.Input.SparseBinary -> LayerFactory(
        inputArrays = inputConfiguration.sizes.map { AugmentedArray<SparseBinaryNDArray>(size = it) },
        inputType = inputConfiguration.type,
        outputSize = outputConfiguration.size,
        params = params,
        activationFunction = outputConfiguration.activationFunction,
        connectionType = outputConfiguration.connectionType,
        dropout = inputConfiguration.dropout,
        contextWindow = contextWindow)
    }
  }

  /**
   * Layer factory used to concatenate two layers, given the input array (referenced from
   * the previous layer) and the output layersConfiguration.
   *
   * @param inputArrays a list of AugmentedArrays used as referenced input (to concatenate two layers)
   * @param inputType the input type
   * @param outputConfiguration the layersConfiguration of the output array
   * @param params the parameters of the layer
   * @param dropout the probability of dropout
   *
   * @return a new Layer
   */
  operator fun <InputNDArrayType : NDArray<InputNDArrayType>> invoke(
    inputArrays: List<AugmentedArray<InputNDArrayType>>,
    inputType: LayerType.Input,
    outputConfiguration: LayerInterface,
    params: LayerParameters,
    dropout: Double,
    contextWindow: LayerContextWindow? = null
  ) : Layer<InputNDArrayType> = LayerFactory(
    inputArrays = inputArrays,
    inputType = inputType,
    outputSize = outputConfiguration.size,
    params = params,
    activationFunction = outputConfiguration.activationFunction,
    connectionType = outputConfiguration.connectionType!!,
    dropout = dropout,
    contextWindow = contextWindow)

  /**
   * Build a new generic [Layer].
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
   * @return a new layer
   */
  operator fun <InputNDArrayType : NDArray<InputNDArrayType>> invoke(
    inputArrays: List<AugmentedArray<InputNDArrayType>>,
    inputType: LayerType.Input,
    outputSize: Int,
    params: LayerParameters,
    connectionType: LayerType.Connection,
    activationFunction: ActivationFunction? = null,
    dropout: Double = 0.0,
    contextWindow: LayerContextWindow? = null
  ) : Layer<InputNDArrayType> = when (connectionType) {

    LayerType.Connection.Feedforward -> FeedforwardLayer(
      inputArray = inputArrays.first(),
      inputType = inputType,
      outputArray = AugmentedArray.zeros(outputSize),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout)

    LayerType.Connection.Highway -> HighwayLayer(
      inputArray = inputArrays.first(),
      inputType = inputType,
      outputArray = AugmentedArray.zeros(outputSize),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout)

    LayerType.Connection.SquaredDistance -> SquaredDistanceLayer(
      inputArray = inputArrays.first(),
      inputType = inputType,
      outputArray = AugmentedArray.zeros(outputSize),
      params = params as SquaredDistanceLayerParameters,
      dropout = dropout)

    LayerType.Connection.Affine -> AffineLayer(
      inputArrays = inputArrays,
      inputType = inputType,
      outputArray = AugmentedArray.zeros(outputSize),
      params = params as AffineLayerParameters,
      activationFunction = activationFunction,
      dropout = dropout)

    LayerType.Connection.Biaffine -> BiaffineLayer(
      inputArray1 = inputArrays[0],
      inputArray2 = inputArrays[1],
      inputType = inputType,
      outputArray = AugmentedArray.zeros(outputSize),
      params = params as BiaffineLayerParameters,
      activationFunction = activationFunction,
      dropout = dropout)

    LayerType.Connection.Concat -> ConcatLayer(
      inputArrays = inputArrays,
      inputType = inputType,
      outputArray = AugmentedArray.zeros(outputSize),
      params = params as ConcatLayerParameters)

    LayerType.Connection.Sum -> SumLayer(
      inputArrays = inputArrays,
      inputType = inputType,
      outputArray = AugmentedArray.zeros(outputSize),
      params = params as SumLayerParameters)

    LayerType.Connection.Sub -> SubLayer(
      inputArray1 = inputArrays[0],
      inputArray2 = inputArrays[1],
      inputType = inputType,
      outputArray = AugmentedArray.zeros(outputSize),
      params = params as SubLayerParameters)

    LayerType.Connection.Avg -> AvgLayer(
      inputArrays = inputArrays,
      inputType = inputType,
      outputArray = AugmentedArray.zeros(outputSize),
      params = params as AvgLayerParameters)

    LayerType.Connection.Product -> ProductLayer(
      inputArrays = inputArrays,
      inputType = inputType,
      outputArray = AugmentedArray.zeros(outputSize),
      params = params as ProductLayerParameters)

    LayerType.Connection.SimpleRecurrent -> SimpleRecurrentLayer(
      inputArray = inputArrays.first(),
      inputType = inputType,
      outputArray = RecurrentLayerUnit(outputSize),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!)

    LayerType.Connection.GRU -> GRULayer(
      inputArray = inputArrays.first(),
      inputType = inputType,
      outputArray = AugmentedArray.zeros(outputSize),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!)

    LayerType.Connection.LSTM -> LSTMLayer(
      inputArray = inputArrays.first(),
      inputType = inputType,
      outputArray = AugmentedArray.zeros(outputSize),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!)

    LayerType.Connection.CFN -> CFNLayer(
      inputArray = inputArrays.first(),
      inputType = inputType,
      outputArray = AugmentedArray.zeros(outputSize),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!)

    LayerType.Connection.RAN -> RANLayer(
      inputArray = inputArrays.first(),
      inputType = inputType,
      outputArray = AugmentedArray.zeros(outputSize),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!)

    LayerType.Connection.DeltaRNN -> DeltaRNNLayer(
      inputArray = inputArrays.first(),
      inputType = inputType,
      outputArray = AugmentedArray.zeros(outputSize),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!)

    LayerType.Connection.IndRNN -> IndRNNLayer(
      inputArray = inputArrays.first(),
      inputType = inputType,
      outputArray = AugmentedArray.zeros(outputSize),
      params = params,
      activationFunction = activationFunction,
      dropout = dropout,
      layerContextWindow = contextWindow!!)

    LayerType.Connection.LTM -> LTMLayer(
      inputArray = inputArrays.first(),
      inputType = inputType,
      outputArray = AugmentedArray.zeros(outputSize),
      params = params,
      dropout = dropout,
      layerContextWindow = contextWindow!!)

    LayerType.Connection.TPR -> TPRLayer(
      inputArray = inputArrays.first(),
      inputType = inputType,
      params = params,
      dropout = dropout,
      layerContextWindow = contextWindow!!,
      q = 0.00001) // TODO
  }
}
