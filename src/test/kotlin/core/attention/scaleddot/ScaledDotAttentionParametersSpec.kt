/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.attention.scaleddot

import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.models.attention.scaleddot.ScaledDotAttentionLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class ScaledDotAttentionParametersSpec : Spek({

  describe("ScaledDotAttentionParameters") {

    context("initialization") {

      val randomGenerator = mock<RandomGenerator>()
      var i = 0.0
      whenever(randomGenerator.next()).thenAnswer { i++ }

      val params = ScaledDotAttentionLayerParameters(
        inputSize = 2,
        attentionSize = 3,
        outputSize = 2,
        weightsInitializer = RandomInitializer(randomGenerator),
        biasesInitializer = RandomInitializer(randomGenerator))

      it("should have the queries weights with the expected initialized values") {
        assertTrue {
          params.queries.weights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.0, 3.0),
              doubleArrayOf(1.0, 4.0),
              doubleArrayOf(2.0, 5.0)
            )),
            tolerance = 1.0e-06
          )
        }
      }

      it("should have the keys weights with the expected initialized values") {
        assertTrue {
          params.keys.weights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(6.0, 9.0),
              doubleArrayOf(7.0, 10.0),
              doubleArrayOf(8.0, 11.0)
            )),
            tolerance = 1.0e-06
          )
        }
      }

      it("should have the values weights with the expected initialized values") {
        assertTrue {
          params.values.weights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(12.0, 14.0),
              doubleArrayOf(13.0, 15.0)
            )),
            tolerance = 1.0e-06
          )
        }
      }

      it("should have the queries biases with the expected initialized values") {
        assertTrue {
          params.queries.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(16.0, 17.0, 18.0)),
            tolerance = 1.0e-06)
        }
      }

      it("should have the keys biases with the expected initialized values") {
        assertTrue {
          params.keys.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(19.0, 20.0, 21.0)),
            tolerance = 1.0e-06)
        }
      }

      it("should have the values biases with the expected initialized values") {
        assertTrue {
          params.values.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(22.0, 23.0)),
            tolerance = 1.0e-06)
        }
      }
    }
  }
})
