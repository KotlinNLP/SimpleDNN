/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.cfn.CFNLayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.deltarnn.DeltaRNNLayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.gru.GRULayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.lstm.LSTMLayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.ran.RANLayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.simple.SimpleRecurrentLayerParameters


/**
 *
 */
object LayerParametersFactory {

  operator fun invoke(inputSize: Int,
                      outputSize: Int,
                      connectionType: LayerType.Connection,
                      weightsInitializer: Initializer?,
                      biasesInitializer: Initializer?,
                      sparseInput: Boolean = false,
                      meProp: Boolean = false): LayerParameters<*> = when(connectionType) {

    LayerType.Connection.Feedforward -> FeedforwardLayerParameters(
      inputSize = inputSize,
      outputSize = outputSize,
      sparseInput = sparseInput,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer,
      meProp = meProp)

    LayerType.Connection.SimpleRecurrent -> SimpleRecurrentLayerParameters(
      inputSize = inputSize,
      outputSize = outputSize,
      sparseInput = sparseInput,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer) // TODO: set 'meProp' param

    LayerType.Connection.GRU -> GRULayerParameters(
      inputSize = inputSize,
      outputSize = outputSize,
      sparseInput = sparseInput,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer) // TODO: set 'meProp' param

    LayerType.Connection.LSTM -> LSTMLayerParameters(
      inputSize = inputSize,
      outputSize = outputSize,
      sparseInput = sparseInput,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer) // TODO: set 'meProp' param

    LayerType.Connection.CFN -> CFNLayerParameters(
      inputSize = inputSize,
      outputSize = outputSize,
      sparseInput = sparseInput,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer) // TODO: set 'meProp' param

    LayerType.Connection.RAN -> RANLayerParameters(
      inputSize = inputSize,
      outputSize = outputSize,
      sparseInput = sparseInput,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer,
      meProp = meProp)

    LayerType.Connection.DeltaRNN -> DeltaRNNLayerParameters(
      inputSize = inputSize,
      outputSize = outputSize,
      sparseInput = sparseInput,
      weightsInitializer = weightsInitializer,
      biasesInitializer = biasesInitializer) // TODO: set 'meProp' param
  }
}
