/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package neuralnetwork

import com.kotlinnlp.simplednn.core.functionalities.activations.*
import com.kotlinnlp.simplednn.core.layers.LayerConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerStructure
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.structure.feedforward.FeedforwardNetworkStructure
import com.kotlinnlp.simplednn.simplemath.NDArray
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import layers.structure.utils.FeedforwardLayerStructureUtils
import neuralnetwork.utils.FeedforwardNetworkStructureUtils
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals
import kotlin.test.assertFails

/**
 *
 */
class FeedforwardNetworkStructureSpec : Spek({

  describe("a FeedforwardNetworkStructure") {

    context("invalid configurations") {

      on("initialization with null output connection types") {

        val layersConfiguration = arrayOf(
          LayerConfiguration(size = 4),
          LayerConfiguration(size = 5, activationFunction = Tanh()),
          LayerConfiguration(size = 3, activationFunction = Softmax(), connectionType = LayerType.Connection.Feedforward)
        ).toList()

        it("should throw an exception") {
          assertFails {
            FeedforwardNetworkStructure(
              layersConfiguration = layersConfiguration,
              params = FeedforwardNetworkStructureUtils.buildParams(layersConfiguration))
          }
        }
      }

      on("initialization with connection types not allowed") {

        val layersConfiguration = arrayOf(
          LayerConfiguration(size = 4),
          LayerConfiguration(size = 5, activationFunction = Tanh(), connectionType = LayerType.Connection.GRU),
          LayerConfiguration(size = 3, activationFunction = Softmax(), connectionType = LayerType.Connection.Feedforward)
        ).toList()

        it("should throw an exception") {
          assertFails {
            FeedforwardNetworkStructure(
              layersConfiguration = layersConfiguration,
              params = FeedforwardNetworkStructureUtils.buildParams(layersConfiguration))
          }
        }
      }
    }

    context("correct configuration") {

      val layersConfiguration = arrayOf(
        LayerConfiguration(size = 4),
        LayerConfiguration(size = 5, activationFunction = Tanh(), connectionType = LayerType.Connection.Feedforward),
        LayerConfiguration(size = 3, activationFunction = Softmax(), connectionType = LayerType.Connection.Feedforward)
      ).toList()

      val structure = FeedforwardNetworkStructure(
        layersConfiguration = layersConfiguration,
        params = FeedforwardNetworkStructureUtils.buildParams(layersConfiguration))

      on("architecture") {

        it("should have the expected number of layers") {
          assertEquals(true, structure.layers.size == 2)
        }

        it("should have interconnected layers") {
          for (i in 0 until structure.layers.size - 1) {
            assertEquals(true, structure.layers[i].outputArray == structure.layers[i + 1].inputArray)
          }
        }

        it("should contain the expected input layer") {
          assertEquals(true, structure.inputLayer == structure.layers[0])
        }

        it("should contain the expected output layer") {
          assertEquals(true, structure.outputLayer== structure.layers[1])
        }
      }

      on("layers factory") {

        it("should contain layers of the expected type") {
          structure.layers.forEach { assertEquals(true, it is FeedforwardLayerStructure) }
        }
      }

      on("methods usage") {

        val features = NDArray.arrayOf(doubleArrayOf(-0.8, -0.9, -0.9, 1.0))
        val output = structure.forward(features)
        val expectedOutput = NDArray.arrayOf(doubleArrayOf(0.19, 0.29, 0.53))

        it("should return the expected output after a call of the forward method") {
          assertEquals(true, output.equals(expectedOutput, tolerance = 0.005))
        }

        val outputGold = FeedforwardLayerStructureUtils.getOutputGold3()
        val errors = MSECalculator().calculateErrors(
          output = structure.outputLayer.outputArray.values,
          outputGold = outputGold)

        structure.backward(
          outputErrors = errors,
          paramsErrors = NetworkParameters(layersConfiguration),
          propagateToInput = true)

        val inputErrors = structure.inputLayer.inputArray.errors
        val expectedInputErrors = NDArray.arrayOf(doubleArrayOf(0.32, -0.14, -0.06, 0.07))

        it("should contain the expected input error after a call of the backward method") {
          assertEquals(true, inputErrors.equals(expectedInputErrors, tolerance = 0.005))
        }
      }
    }
  }
})
