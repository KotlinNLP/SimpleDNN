/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.neuralnetwork

import com.kotlinnlp.simplednn.core.functionalities.initializers.ConstantInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import core.neuralnetwork.utils.SerializedNetwork
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import java.io.*
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 *
 */
class NeuralNetworkSpec: Spek({

  describe("a NeuralNetwork") {

    context("loading from a byte array input stream") {

      it("should load a NeuralNetwork without failing") {
        StackedLayersParameters.load(inputStream = ByteArrayInputStream(SerializedNetwork.byteArray))
      }
    }

    context("dumping to a byte array output stream") {

      val network = StackedLayersParameters(
        LayerInterface(size = 3),
        LayerInterface(size = 5, connectionType = LayerType.Connection.Feedforward)
      )

      val outputStream = ByteArrayOutputStream()

      network.dump(outputStream)

//      outputStream.toByteArray().forEachIndexed { i, b ->
//        print("%d, ".format(b))
//        if ((i + 1) % 20 == 0) print("\n")
//      }
//      print("\n")

      it("should write to the output stream") {
        assertTrue { outputStream.size() > 0 }
      }
    }

    context("initialization") {

      var k = 0
      val initValues = doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
      val randomGenerator = mock<RandomGenerator>()
      whenever(randomGenerator.next()).thenAnswer { initValues[k++] }

      val network = StackedLayersParameters(
        LayerInterface(size = 3),
        LayerInterface(size = 2, connectionType = LayerType.Connection.Feedforward),
        weightsInitializer = RandomInitializer(randomGenerator),
        biasesInitializer = ConstantInitializer(0.9)
      )

      val params = network.paramsPerLayer[0] as FeedforwardLayerParameters
      val w = params.unit.weights.values
      val b = params.unit.biases.values

      it("should contain the expected initialized weights") {
        (0 until w.length).forEach { i -> assertEquals(initValues[i], w[i]) }
      }

      it("should contain the expected initialized biases") {
        (0 until b.length).forEach { i -> assertEquals(0.9, b[i]) }
      }
    }
  }
})
