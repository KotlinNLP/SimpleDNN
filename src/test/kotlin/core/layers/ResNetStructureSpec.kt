/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, you can obtain one at http://mozilla.org/MPL/2.0/.
* ------------------------------------------------------------------*/

package core.layers

import com.kotlinnlp.simplednn.core.functionalities.activations.ReLU
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.layers.*
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import core.layers.feedforward.simple.FeedforwardLayerStructureUtils
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ResNetStructureSpec : Spek({

    describe("a ResNet") {

      context("sumLayer not the same size of input") {

        val layersConfiguration = arrayOf(
            LayerInterface(size = 4),
            LayerInterface(size = 5, activationFunction = ReLU(), connectionType = LayerType.Connection.Feedforward),
            LayerInterface(size = 3, activationFunction = null, connectionType = LayerType.Connection.Feedforward)
        ).toList()

        val structure = ResNet<DenseNDArray>(
            layersConfiguration = layersConfiguration,
            paramsPerLayer = ResNetStructureUtils.buildParams(layersConfiguration).paramsPerLayer,
            sumFeedForwardParams = ResNetStructureUtils.getParams43(),
            outputActivation = ReLU())

        on("architecture") {

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

        on("layers factory") {

          it("should contain layers of the expected type") {
            structure.layers.forEach { assertTrue { it is FeedforwardLayer } }
          }
        }

        on("methods usage") {

          val features = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.9, -0.9, 1.0))
          val output = structure.forward(features)
          val expectedOutput = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 1.502, 0.319))

          it("should return the expected output after a call of the forward method") {
            assertTrue { output.equals(expectedOutput, tolerance = 0.005) }
          }

          val outputGold = FeedforwardLayerStructureUtils.getOutputGold3()

          structure.backward(
              outputErrors = structure.outputLayer.outputArray.values.sub(outputGold),
              propagateToInput = true)

          val inputErrors = structure.inputLayer.inputArray.errors
          val expectedInputErrors = DenseNDArrayFactory.arrayOf(doubleArrayOf(3.02247, -2.71622, -0.16097, 1.2028))

          it("should contain the expected input error after a call of the backward method") {
            assertTrue { inputErrors.equals(expectedInputErrors, tolerance = 0.005) }
          }
        }
      }
    }
  })
