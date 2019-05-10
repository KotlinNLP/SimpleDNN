/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.feedforward.highway.HighwayLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.cfn.CFNLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.deltarnn.DeltaRNNLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.gru.GRULayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.indrnn.IndRNNLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.lstm.LSTMLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.ran.RANLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.affine.AffineLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.avg.AvgLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.biaffine.BiaffineLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.concat.ConcatLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.product.ProductLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.sum.SumLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.ltm.LTMLayerParameters

/**
 * Helper that builds generic [LayerParameters].
 */
object LayerParametersFactory {

  /**
   * Build new generic [LayerParameters].
   *
   * @param inputsSize the list of input sizes (more then one only for Merge layers)
   * @param outputSize the size of the output (null for layers with fixed output)
   * @param connectionType the type of connection from the input to the output
   * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
   * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
   * @param sparseInput whether the weights connected to the input are sparse or not
   *
   * @return new layer parameters
   */
  operator fun invoke(inputsSize: List<Int>,
                      outputSize: Int? = null,
                      connectionType: LayerType.Connection,
                      weightsInitializer: Initializer?,
                      biasesInitializer: Initializer?,
                      sparseInput: Boolean = false): LayerParameters<*> = when (connectionType) {

    LayerType.Connection.Feedforward -> FeedforwardLayerParameters(
      inputSize = inputsSize.first(),
      outputSize = outputSize!!,
      sparseInput = sparseInput,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer)

    LayerType.Connection.Highway -> {

      require(outputSize == null || outputSize == inputsSize.first()) {
        "The Highway layer requires that the output size must be equal to the input size."
      }

      HighwayLayerParameters(
        inputSize = inputsSize.first(),
        sparseInput = sparseInput,
        weightsInitializer = weightsInitializer,
        biasesInitializer = biasesInitializer)
    }

    LayerType.Connection.Affine -> AffineLayerParameters(
      inputsSize = inputsSize,
      outputSize = outputSize!!,
      sparseInput = sparseInput,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer)

    LayerType.Connection.Biaffine -> BiaffineLayerParameters(
      inputSize1 = inputsSize[0],
      inputSize2 = inputsSize[1],
      outputSize = outputSize!!,
      sparseInput = sparseInput,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer)

    LayerType.Connection.Concat -> {

      require(outputSize == null || outputSize == inputsSize.sum()) {
        "The Concat merge layer requires that the output size must double the input size."
      }

      ConcatLayerParameters(inputsSize = inputsSize)
    }

    LayerType.Connection.Sum -> {

      require(outputSize == null || outputSize == inputsSize.first()) {
        "The Sum merge layer requires that the output size must be equal to the input size."
      }

      SumLayerParameters(inputSize = inputsSize.first(), nInputs = inputsSize.size)
    }

    LayerType.Connection.Avg -> {

      require(outputSize == null || outputSize == inputsSize.first()) {
        "The Avg merge layer requires that the output size must be equal to the input size."
      }

      AvgLayerParameters(inputSize = inputsSize.first(), nInputs = inputsSize.size)
    }

    LayerType.Connection.Product -> {

      require(outputSize == null || outputSize == inputsSize.first()) {
        "The Product merge layer requires that the output size must be equal to the input size."
      }

      ProductLayerParameters(inputSize = inputsSize.first(), nInputs = inputsSize.size)
    }

    LayerType.Connection.SimpleRecurrent -> SimpleRecurrentLayerParameters(
      inputSize = inputsSize.first(),
      outputSize = outputSize!!,
      sparseInput = sparseInput,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer)

    LayerType.Connection.GRU -> GRULayerParameters(
      inputSize = inputsSize.first(),
      outputSize = outputSize!!,
      sparseInput = sparseInput,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer)

    LayerType.Connection.LSTM -> LSTMLayerParameters(
      inputSize = inputsSize.first(),
      outputSize = outputSize!!,
      sparseInput = sparseInput,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer)

    LayerType.Connection.CFN -> CFNLayerParameters(
      inputSize = inputsSize.first(),
      outputSize = outputSize!!,
      sparseInput = sparseInput,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer)

    LayerType.Connection.RAN -> RANLayerParameters(
      inputSize = inputsSize.first(),
      outputSize = outputSize!!,
      sparseInput = sparseInput,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer)

    LayerType.Connection.DeltaRNN -> DeltaRNNLayerParameters(
      inputSize = inputsSize.first(),
      outputSize = outputSize!!,
      sparseInput = sparseInput,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer)

    LayerType.Connection.IndRNN -> IndRNNLayerParameters(
      inputSize = inputsSize.first(),
      outputSize = outputSize!!,
      sparseInput = sparseInput,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer)

    LayerType.Connection.LTM -> {

      require(outputSize == null || outputSize == inputsSize.first()) {
        "The LTM layer requires that the output size must be equal to the input size."
      }

      LTMLayerParameters(
        inputSize = inputsSize.first(),
        sparseInput = sparseInput,
        weightsInitializer = weightsInitializer,
        biasesInitializer = biasesInitializer)
    }
  }
}
