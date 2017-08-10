/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package neuralnetwork

import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.LayerConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import neuralnetwork.utils.SerializedNetwork
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import java.io.*
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 *
 */
class NeuralNetworkSpec: Spek({

  describe("a NeuralNetwork") {

    on("loading from a byte array input stream") {

      val inputStream = ByteArrayInputStream(SerializedNetwork.byteArray)
      val network = NeuralNetwork.load(inputStream)

      it("should return a NeuralNetwork") {
        assertTrue { network is NeuralNetwork }
      }
    }

    on("dumping to a byte array output stream") {

      val network = NeuralNetwork(
        LayerConfiguration(size = 3),
        LayerConfiguration(size = 5, connectionType = LayerType.Connection.Feedforward)
      )

      val outputStream = ByteArrayOutputStream()

      network.dump(outputStream)

//      outputStream.toByteArray().forEachIndexed { i, b ->
//        print("%d, ".format(b))
//        if ((i + 1) % 20 == 0) print("\n")
//      }

      it("should write to the output stream") {
        assertTrue { outputStream.size() > 0 }
      }
    }

    on("initialization") {

      val network = NeuralNetwork(
        LayerConfiguration(size = 3),
        LayerConfiguration(size = 2, connectionType = LayerType.Connection.Feedforward)
      )

      var k = 0
      val initValues = doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
      val randomGenerator = mock<RandomGenerator>()
      whenever(randomGenerator.next()).thenAnswer { initValues[k++] }

      network.initialize(randomGenerator = randomGenerator, biasesInitValue = 0.9)

      val params = network.model.paramsPerLayer[0] as FeedforwardLayerParameters
      val w = params.unit.weights.values
      val b = params.unit.biases.values

      it("should contain the expected initialized weights") {
        (0 until w.length).forEach({ i -> assertEquals(initValues[i], w[i]) })
      }

      it("should contain the expected initialized biases") {
        (0 until b.length).forEach({ i -> assertEquals(0.9, b[i]) })
      }
    }
  }
})
