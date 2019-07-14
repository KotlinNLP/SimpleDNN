/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.feedforward.normalization

import com.kotlinnlp.simplednn.core.functionalities.initializers.ConstantInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.models.feedforward.normalization.NormalizationLayerParameters
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class NormalizationLayerParametersSpec: Spek({

  describe("a ConvolutionLayerParameters") {

    context("initialization") {

      context("dense input") {

        var k = 0
        val initValues = doubleArrayOf(
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
            1.2, 1.3, 1.4, 1.5, 1.6)
        val randomGenerator = mock<RandomGenerator>()
        whenever(randomGenerator.next()).thenAnswer { initValues[k++] }

        val params = NormalizationLayerParameters(
            inputSize = 10,
            weightsInitializer = RandomInitializer(randomGenerator),
            biasesInitializer = ConstantInitializer(0.9))

        val s = params.paramsList[0].values
        val b = params.paramsList[1].values

        it("parameters should contain the expected number of vectors/matrices") {
          assertEquals(2, params.paramsList.size)
        }

        it("should get the expected param at index 0") {
          assertTrue { s === params[0].values }
        }

        it("should get the expected param at index 1") {
          assertTrue { b === params[1].values }
        }

        it("should contain the expected initialized w1") {
          (0 until s.length).forEach { i -> assertEquals(initValues[i], s[i]) }
        }

        it("should contain the expected initialized biases") {
          (0 until b.length).forEach { i -> assertEquals(0.9, b[i]) }
        }

      }
    }
  }
})