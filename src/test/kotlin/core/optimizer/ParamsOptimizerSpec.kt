/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.optimizer

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateMethod
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class ParamsOptimizerSpec : Spek({

  describe("a ParamsOptimizer") {

    val learningRateMethod = LearningRateMethod(learningRate = 0.1)

    context("update after accumulate") {

      val optimizer = ParamsOptimizer(learningRateMethod)

      val params: FeedforwardLayerParameters = ParamsOptimizerUtils.buildParams()

      val gw1 = params.unit.weights.buildDenseErrors(ParamsOptimizerUtils.buildWeightsErrorsValues1())
      val gb1 = params.unit.biases.buildDenseErrors(ParamsOptimizerUtils.buildBiasesErrorsValues1())
      val gw2 = params.unit.weights.buildDenseErrors(ParamsOptimizerUtils.buildWeightsErrorsValues2())
      val gb2 = params.unit.biases.buildDenseErrors(ParamsOptimizerUtils.buildBiasesErrorsValues2())

      optimizer.accumulate(listOf(gw1, gb1, gw2, gb2))
      optimizer.update()

      val w: DenseNDArray = params[0].values
      val b: DenseNDArray = params[1].values

      it("should match the expected updated weights") {
        assertTrue {
          w.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.25, 0.42, 0.185, -0.16),
              doubleArrayOf(0.15, -0.125, 0.14, 0.58)
            )),
            tolerance = 1.0e-06
          )
        }
      }

      it("should match the expected updated biases") {
        assertTrue {
          b.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.33, -0.385)),
            tolerance = 1.0e-06
          )
        }
      }
    }
  }
})
