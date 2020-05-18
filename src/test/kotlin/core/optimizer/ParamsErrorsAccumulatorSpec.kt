/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.optimizer

import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class ParamsErrorsAccumulatorSpec : Spek({

  describe("a ParamsErrorsAccumulator") {

    context("initialization") {

      val accumulator = ParamsErrorsAccumulator()

      it("should return an empty list before accumulation") {
        assertTrue { accumulator.getParamsErrors().isEmpty() }
      }
    }

    context("accumulation") {

      val accumulator = ParamsErrorsAccumulator()

      val params: FeedforwardLayerParameters = ParamsErrorsAccumulatorUtils.buildEmptyParams()

      val gw1 = params.unit.weights.buildDenseErrors(ParamsErrorsAccumulatorUtils.buildWeightsErrorsValues1())
      val gb1 = params.unit.biases.buildDenseErrors(ParamsErrorsAccumulatorUtils.buildBiasesErrorsValues1())
      val gw2 = params.unit.weights.buildDenseErrors(ParamsErrorsAccumulatorUtils.buildWeightsErrorsValues2())
      val gb2 = params.unit.biases.buildDenseErrors(ParamsErrorsAccumulatorUtils.buildBiasesErrorsValues2())

      accumulator.accumulate(listOf(gw1, gb1, gw2, gb2))

      val accumulatedErrors = accumulator.getParamsErrors()
      val accW: DenseNDArray = accumulatedErrors[0].values as DenseNDArray
      val accB: DenseNDArray = accumulatedErrors[1].values as DenseNDArray

      it("should match the expected accumulated weights errors") {
        assertTrue {
          accW.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(1.0, -0.4, 0.3, -0.8),
              doubleArrayOf(1.0, 0.5, -0.8, 0.4)
            )),
            tolerance = 1.0e-06
          )
        }
      }

      it("should match the expected accumulated biases errors") {
        assertTrue {
          accB.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.6, -0.3)),
            tolerance = 1.0e-06
          )
        }
      }

      accumulator.clear()

      it("should return an empty list of errors when calling getParamsErrors after a clear()") {
        assertTrue { accumulator.getParamsErrors().isEmpty() }
      }
    }
  }
})
