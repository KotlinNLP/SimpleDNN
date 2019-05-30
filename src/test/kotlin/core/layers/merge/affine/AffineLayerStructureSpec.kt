/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain contexte at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.merge.affine

import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class AffineLayerStructureSpec : Spek({

  describe("an AffineLayer") {

    context("forward") {

      val layer = AffineLayerUtils.buildLayer()
      layer.forward()

      it("should match the expected outputArray") {
        assertTrue {
          layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.664037, -0.019997)),
            tolerance = 1.0e-06)
        }
      }
    }

    context("backward") {

      val layer = AffineLayerUtils.buildLayer()

      layer.forward()

      layer.outputArray.assignErrors(layer.outputArray.values.sub(AffineLayerUtils.getOutputGold()))
      val paramsErrors = layer.backward(propagateToInput = true)

      val params = layer.params

      it("should match the expected errors of the outputArray") {
        assertTrue {
          layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.427139, 0.279891)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the biases") {
        assertTrue {
          paramsErrors.getErrorsOf(params.b)!!.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.427139, 0.279891)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of w1") {
        assertTrue {
          (paramsErrors.getErrorsOf(params.w[0])!!.values).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.341711, 0.384425),
              doubleArrayOf(-0.223913, -0.251902)
            )),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of w2") {
        assertTrue {
          (paramsErrors.getErrorsOf(params.w[1])!!.values).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.213569, 0.085428, -0.256283),
              doubleArrayOf(0.139945, -0.055978, 0.167934)
            )),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the inputArray1") {
        assertTrue {
          layer.inputArrays[0].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.095771, -0.537634)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the inputArray2") {
        assertTrue {
          layer.inputArrays[1].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.172316, -0.297537, 0.468392)),
            tolerance = 1.0e-06)
        }
      }
    }
  }
})
