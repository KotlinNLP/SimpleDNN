/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.neuralnetwork

import com.kotlinnlp.simplednn.core.functionalities.activations.*
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType.Connection
import com.kotlinnlp.simplednn.core.layers.StackedLayers
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayer
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import core.layers.feedforward.simple.FeedforwardLayerStructureUtils
import core.neuralnetwork.utils.FeedforwardNetworkStructureUtils
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 *
 */
class FeedforwardNetworkStructureSpec : Spek({

  describe("a FeedforwardStackedLayers") {

    context("invalid configurations") {

      context("initialization with null output connection types") {

        val wrongLayersConfiguration = arrayOf(
          LayerInterface(size = 4),
          LayerInterface(size = 5, activationFunction = Tanh()),
          LayerInterface(size = 3, activationFunction = Softmax(), connectionType = Connection.Feedforward)
        ).toList()

        it("should throw an exception") {
          assertFailsWith<KotlinNullPointerException> {
            StackedLayers<DenseNDArray>(params = StackedLayersParameters(wrongLayersConfiguration))
          }
        }
      }
    }

    context("correct configuration") {

      val layersConfiguration = arrayOf(
        LayerInterface(size = 4),
        LayerInterface(size = 5, activationFunction = Tanh(), connectionType = Connection.Feedforward),
        LayerInterface(size = 3, activationFunction = Softmax(), connectionType = Connection.Feedforward)
      ).toList()

      val structure = StackedLayers<DenseNDArray>(
        params = FeedforwardNetworkStructureUtils.buildParams(layersConfiguration))

      context("architecture") {

        it("should have the expected number of layers") {
          assertEquals(2, structure.layers.size)
        }

        it("should have interconnected layers") {
          for (i in 0 until structure.layers.size - 1) {
            assertEquals(structure.layers[i].outputArray, structure.layers[i + 1].inputArray)
          }
        }

        it("should contain the expected input layer") {
          assertEquals(structure.inputLayer, structure.layers[0])
        }

        it("should contain the expected output layer") {
          assertEquals(structure.outputLayer, structure.layers[1])
        }
      }

      context("layers factory") {

        it("should contain layers of the expected type") {
          structure.layers.forEach { assertTrue { it is FeedforwardLayer } }
        }
      }

      context("methods usage") {

        val features = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.9, -0.9, 1.0))
        val output = structure.forward(features)
        val expectedOutput = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.19, 0.29, 0.53))

        it("should return the expected output after a call of the forward method") {
          assertTrue { output.equals(expectedOutput, tolerance = 0.005) }
        }

        val outputGold = FeedforwardLayerStructureUtils.getOutputGold3()

        structure.backward(
          outputErrors = structure.outputLayer.outputArray.values.sub(outputGold),
          propagateToInput = true)

        val inputErrors = structure.inputLayer.inputArray.errors
        val expectedInputErrors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.32, -0.14, -0.06, 0.07))

        it("should contain the expected input error after a call of the backward method") {
          assertTrue { inputErrors.equals(expectedInputErrors, tolerance = 0.005) }
        }
      }
    }
  }
})
