/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.attention

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.attention.han.HAN
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 *
 */
class HANSpec : Spek({

  describe("a HAN") {

    context("initialization") {

      val han = HAN(
        hierarchySize = 3,
        inputSize = 10,
        biRNNsActivation = Tanh(),
        biRNNsConnectionType = LayerType.Connection.GRU,
        attentionSize = 5,
        outputSize = 30,
        outputActivation = Softmax(),
        gainFactors = listOf(1.5, 2.0, 1.0))

      it("should have a BiRNN with the expected input size in the level 2") {
        assertEquals(10, han.biRNNs[2].inputSize)
      }

      it("should have a BiRNN with the expected output size in the level 2") {
        assertEquals(16, han.biRNNs[2].outputSize)
      }

      it("should have a BiRNN with the expected activation function in the level 2") {
        assertTrue { han.biRNNs[2].hiddenActivation is Tanh }
      }

      it("should have AttentionNetworkParams with the expected input size in the level 2") {
        assertEquals(16, han.attentionNetworksParams[2].inputSize)
      }

      it("should have AttentionNetworkParams with the expected output size in the level 2") {
        assertEquals(16, han.attentionNetworksParams[2].outputSize)
      }

      it("should have AttentionNetworkParams with the expected attention size in the level 2") {
        assertEquals(5, han.attentionNetworksParams[2].attentionSize)
      }

      it("should have a BiRNN with the expected input size in the level 1") {
        assertEquals(16, han.biRNNs[1].inputSize)
      }

      it("should have a BiRNN with the expected output size in the level 1") {
        assertEquals(32, han.biRNNs[1].outputSize)
      }

      it("should have a BiRNN with the expected activation function in the level 1") {
        assertTrue { han.biRNNs[1].hiddenActivation is Tanh }
      }

      it("should have AttentionNetworkParams with the expected input size in the level 1") {
        assertEquals(32, han.attentionNetworksParams[1].inputSize)
      }

      it("should have AttentionNetworkParams with the expected output size in the level 1") {
        assertEquals(32, han.attentionNetworksParams[1].outputSize)
      }

      it("should have AttentionNetworkParams with the expected attention size in the level 1") {
        assertEquals(5, han.attentionNetworksParams[1].attentionSize)
      }

      it("should have a BiRNN with the expected input size in the level 0") {
        assertEquals(32, han.biRNNs[0].inputSize)
      }

      it("should have a BiRNN with the expected output size in the level 0") {
        assertEquals(32, han.biRNNs[0].outputSize)
      }

      it("should have a BiRNN with the expected activation function in the level 0") {
        assertTrue { han.biRNNs[0].hiddenActivation is Tanh }
      }

      it("should have AttentionNetworkParams with the expected input size in the level 0") {
        assertEquals(32, han.attentionNetworksParams[0].inputSize)
      }

      it("should have AttentionNetworkParams with the expected output size in the level 0") {
        assertEquals(32, han.attentionNetworksParams[0].outputSize)
      }

      it("should have AttentionNetworkParams with the expected attention size in the level 0") {
        assertEquals(5, han.attentionNetworksParams[0].attentionSize)
      }

      it("should have an output network with the expected input size") {
        assertEquals(32, han.outputNetwork.layersConfiguration[0].size)
      }

      it("should have an output network with the expected output size") {
        assertEquals(30, han.outputNetwork.layersConfiguration[1].size)
      }
    }
  }
})
