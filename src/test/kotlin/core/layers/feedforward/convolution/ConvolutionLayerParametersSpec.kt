/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.feedforward.convolution

import com.kotlinnlp.simplednn.core.functionalities.initializers.ConstantInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.models.feedforward.convolution.ConvolutionLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 *
 */
class ConvolutionLayerParametersSpec : Spek({

  describe("a FeedforwardLayerParameters") {

    context("initialization") {

      on("dense input") {

        var k = 0
        val initValues = doubleArrayOf(
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
            1.2, 1.3, 1.4, 1.5, 1.6)
        val randomGenerator = mock<RandomGenerator>()
        whenever(randomGenerator.next()).thenAnswer { initValues[k++] }

        val params = ConvolutionLayerParameters(
            kernelSize = Shape(2, 2),
            inputChannels = 2,
            outputChannels = 2,
            weightsInitializer = RandomInitializer(randomGenerator),
            biasesInitializer = ConstantInitializer(0.9))

        val w1 = params.paramsList[0].values
        val w2 = params.paramsList[1].values
        val w3 = params.paramsList[2].values
        val w4 = params.paramsList[3].values
        val b1 = params.paramsList[4].values
        val b2 = params.paramsList[5].values

        it("parameters should contain the expected number of vectors/matrices") {
          assertEquals(6, params.paramsList.size)
        }

        it("should contain a dense w1 of expected size") {
          assertTrue { w1.shape.dim1 == 2 && w1.shape.dim2 == 2 }
        }

        it("should contain a dense w2 of expected size") {
          assertTrue { w2.shape.dim1 == 2 && w2.shape.dim2 == 2 }
        }

        it("should contain a dense w3 of expected size") {
          assertTrue { w3.shape.dim1 == 2 && w3.shape.dim2 == 2 }
        }

        it("should contain a dense w4 of expected size") {
          assertTrue { w4.shape.dim1 == 2 && w4.shape.dim2 == 2 }
        }

        it("should contain a dense b1 of expected size") {
          assertTrue { b1.shape.dim1 == 1 && b1.shape.dim2 == 1 }
        }

        it("should contain a dense b1 of expected size") {
          assertTrue {  b1.shape.dim1 == 1 && b1.shape.dim2 == 1  }
        }

        it("should contain the expected initialized w1") {
          (0 until w1.length).forEach { i -> assertEquals(initValues[i], w1[i]) }
        }

        it("should contain the expected initialized w2") {
          (0 until w2.length).forEach { i -> assertEquals(initValues[4 + i], w2[i]) }
        }

        it("should contain the expected initialized w3") {
          (0 until w3.length).forEach { i -> assertEquals(initValues[8 + i], w3[i]) }
        }

        it("should contain the expected initialized w4") {
          (0 until w4.length).forEach { i -> assertEquals(initValues[12 + i], w4[i]) }
        }

        it("should contain the expected initialized biases") {
          (0 until b1.length).forEach { i -> assertEquals(0.9, b1[i]) }
        }

        it("should contain the expected initialized biases") {
          (0 until b2.length).forEach { i -> assertEquals(0.9, b2[i]) }
        }

      }
    }
  }
})